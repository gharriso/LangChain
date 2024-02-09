db.tbj.aggregate(
    [
       {
          $listSearchIndexes:
             {
 
             }
       }
    ]
 )

 db.tbj.createSearchIndex({
    fields: [
        {
          path: 'embedding',
          numDimensions: 1536,
          similarity: 'cosine',
          type: 'vector'
        }
      ]
  }
  )

  db.tbj.createSearchIndex({
    "mappings": {
      "dynamic": true,
      "fields": {
        "plot_embedding": {
          "type": "knnVector",
          "dimensions": 1536,
          "similarity": "euclidean"
        }
      }
    }
  }
  )

  db.tbj.createSearchIndex({
    "mappings": {
      "dynamic": true,
      "fields": {
        "embedding": {
          "type": "vector",
          "dimensions": 1536,
          "similarity": "cosine"
        }
      }
    }
  }
  )

 db.runCommand(
    {
       createSearchIndexes: "tbj",
       indexes: [
          {
             name: "vectorSearchIndex",
             definition: {
                "type": "vectorSearch",

                ]
             }
         },
       ]
    }
 )