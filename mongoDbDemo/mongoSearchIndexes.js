
const database = 'vectorDemo';   
const collection = 'tbj';

// Create a new database.
use(database);

// Create a new collection.
db.createCollection(collection);

 
db.getCollection(collection).createSearchIndex(
  "vector1",
  "vectorSearch",
  {
    fields: [{
      path: "embedding",
      numDimensions: 1536,
      similarity: "cosine",
      type: "vector"
    }]

  }
)

db.getCollection(collection).aggregate(
  [
     {
        $listSearchIndexes:
           {

           }
     }
  ]
)