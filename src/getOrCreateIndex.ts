import { Pinecone } from "@pinecone-database/pinecone";

async function getIndexOrCreate({
  pc,
  indexName,
  dimension,
  metric,
  cloud,
  region,
}: {
  pc: Pinecone;
  indexName: string;
  dimension: number;
  metric: "euclidean" | "cosine" | "dotproduct";
  cloud: "aws" | "gcp" | "azure"; // Ensure valid cloud values
  region: string;
}) {
  try {
    // List all indexes
    const result = await pc.listIndexes();

    // Check if `result.indexes` exists
    if (!result || !result.indexes) {
      // If no indexes exist, create a new index
      const index = await pc.createIndex({
        name: indexName,
        dimension: dimension,
        metric: metric,
        spec: {
          serverless: {
            cloud: cloud,
            region: region,
          },
        },
      });
      return index;
    }

    // Check if the desired index already exists
    const exists = result.indexes.some((idx) => idx.name === indexName);
    if (exists) {
      console.log(`Index "${indexName}" already exists.`);
      return indexName; // Return the name of the existing index
    } else {
      // Create a new index if it doesn't exist
      console.log(`Creating index "${indexName}"...`);
      const index = await pc.createIndex({
        name: indexName,
        dimension: dimension,
        metric: metric,
        spec: {
          serverless: {
            cloud: cloud,
            region: region,
          },
        },
      });
      return index;
    }
  } catch (error) {
    console.error("Error creating or retrieving index:", error);
    throw error;
  }
}

export { getIndexOrCreate };
