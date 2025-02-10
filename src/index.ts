import { GoogleGenerativeAI } from "@google/generative-ai";
import { Pinecone } from "@pinecone-database/pinecone";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { PineconeStore } from "@langchain/pinecone";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

import { getIndexOrCreate } from "./getOrCreateIndex";

import { config } from "dotenv";
config();

const GEMINI_KEY = process.env.GEMINI_API_KEY!;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY!;
const INDEX_NAME = "pdf-index";

const genAI = new GoogleGenerativeAI(GEMINI_KEY);

const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

const pineconeClient = new Pinecone({
  apiKey: PINECONE_API_KEY,
});

let vectorStore;

async function initializeVectorStore() {
  const loader = new PDFLoader("physics.pdf");
  const docs = await loader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 100,
  });

  const splitDocs = await splitter.splitDocuments(docs);

  const embeddings = new GoogleGenerativeAIEmbeddings({
    modelName: "models/embedding-001",
    apiKey: GEMINI_KEY,
  });

  // Get or create the Pinecone index
  const index = await getIndexOrCreate({
    pc: pineconeClient,
    indexName: INDEX_NAME,
    dimension: 768,
    metric: "cosine",
    cloud: "aws",
    region: "us-east-1",
  });

  vectorStore = await PineconeStore.fromDocuments(splitDocs, embeddings, {
    // pineconeIndex: index,
    namespace: "pdf-namespace",
  });

  console.log("PDF processed and stored in database.");
}

async function queryPDF(userQuery: string) {
  try {
    const embeddings = new GoogleGenerativeAIEmbeddings({
      modelName: "models/embedding-001",
      apiKey: GEMINI_KEY,
    });

    const index = await pineconeClient.createIndex({
      name: INDEX_NAME,
      dimension: 768, // Replace with your model dimensions
      metric: "cosine", // Replace with your model metric
      spec: {
        serverless: {
          cloud: "aws",
          region: "ap-northeast-3",
        },
      },
    });

    const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
      //   pineconeIndex: index,
      namespace: "pdf-namespace",
    });

    // Search for relevant documents
    const relevantDocs = await vectorStore.similaritySearch(userQuery, 3);

    // Prepare context from relevant documents
    const context = relevantDocs.map((doc) => doc.pageContent).join("\n\n");

    // Create prompt for Gemini
    const prompt = `
     Context from the PDF:
     ${context}
 
     User Question: ${userQuery}
 
     Please provide a detailed answer based on the context above. If the information is not found in the context, please indicate that.`;

    // Generate response using Gemini
    const result = await model.generateContent(prompt);
    return result.response.text;
  } catch (error) {
    console.error("Error querying PDF:", error);
    throw error;
  }
}

// Main execution function
async function main() {
  try {
    // Initialize the vector store (only need to do this once)
    await initializeVectorStore();
    console.log("Vector store initialized successfully");

    // Example query (you can replace this with actual user queries)
    const query = "What are the main topics discussed in the PDF?";
    const answer = await queryPDF(query);
    console.log("Query:", query);
    console.log("Answer:", answer);
  } catch (error) {
    console.error("Error in main execution:", error);
  }
}

// Run the main function.
main();
