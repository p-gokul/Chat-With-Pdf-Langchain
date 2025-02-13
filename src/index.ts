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
const INDEX_NAME = process.env.INDEX_NAME!;
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL!;
const PDF_NAMESPACE = process.env.PDF_NAMESPACE!;
const FILE_NAME = process.env.FILE_NAME!;
const CHUNK_SIZE = parseInt(process.env.CHUNK_SIZE!);
const CHUNK_OVERLAP = parseInt(process.env.CHUNK_OVERLAP!);
const NO_OF_RELEVANT_CHUNKS = parseInt(process.env.NO_OF_RELEVANT_CHUNKS!);

const genAI = new GoogleGenerativeAI(GEMINI_KEY);

const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

const pineconeClient = new Pinecone({
  apiKey: PINECONE_API_KEY,
});

const embeddings = new GoogleGenerativeAIEmbeddings({
  modelName: EMBEDDING_MODEL,
  apiKey: GEMINI_KEY,
});

const pineconeIndex = pineconeClient.Index(INDEX_NAME);

let vectorStore;

async function initializeVectorStore() {
  // Make sure the index exists, create if not exists
  await getIndexOrCreate({
    pc: pineconeClient,
    indexName: INDEX_NAME, //! sql equivalent -> table
    dimension: 768, // vector size
    metric: "cosine", // distance metric
    cloud: "aws",
    region: "us-east-1",
  });

  const stats = await pineconeIndex.describeIndexStats();

  // If the namespace exists and has at least one vector, assume the PDF has already been processed.
  if (
    stats.namespaces &&
    stats.namespaces[PDF_NAMESPACE] &&
    stats.namespaces[PDF_NAMESPACE].recordCount > 0
  )
    return; // Exit the function early.

  const loader = new PDFLoader(FILE_NAME);
  const docs = await loader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: CHUNK_SIZE,
    chunkOverlap: CHUNK_OVERLAP,
  });

  const splitDocs = await splitter.splitDocuments(docs);

  vectorStore = await PineconeStore.fromDocuments(splitDocs, embeddings, {
    pineconeIndex,
    namespace: "pdf-namespace",
  });

  console.log("PDF processed and stored in database.");
}

async function queryPDF(userQuery: string) {
  try {
    const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
      pineconeIndex,
      namespace: "pdf-namespace",
    });

    // Search for relevant documents
    const relevantDocs = await vectorStore.similaritySearch(
      userQuery,
      NO_OF_RELEVANT_CHUNKS
    );

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
    return result.response.text();
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

    // Example query (you can replace this with actual user queries)
    const query = "Can you explain about the Nuclear Physics.";
    const answer = await queryPDF(query);
    console.log("Query:", query);
    console.log("\n\nAnswer:", answer);
  } catch (error) {
    console.error("Error in main execution:", error);
  }
}

// Run the main function.
main();
