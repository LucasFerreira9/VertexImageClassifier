# 📦 vertex-image-classifier

> An npm package to make it easier to use vertex AI with Google autoML for image classification

<!--
[![npm](https://img.shields.io/npm/v/your-package-name.svg)](https://www.npmjs.com/package/your-package-name)
[![License](https://img.shields.io/npm/l/your-package-name.svg)](./LICENSE)
[![Build](https://img.shields.io/github/actions/workflow/status/your-username/your-repo-name/build.yml?branch=main)](https://github.com/your-username/your-repo-name/actions)
[![Types](https://img.shields.io/npm/types/your-package-name.svg)](https://www.npmjs.com/package/your-package-name)
-->
---

## ✨ Features

- ✅ Integrated with Vertex AI / Google AIplatform
- ⚡ Ideal for classifying images from more complex datasets directly, letting vertex AI algorithms adjust hyperparameters and define the architecture

---

## 📥 Installation


```bash
npm install vertex-image-classifier
# or
yarn add vertex-image-classifier
```

## Usage
```ts
const classifier = new VertexClassifier({
    projectId: "my-project-id",
    location: 'gc-location',
    datasetBucketName: 'my-bucket-name',
    vertexDatasetName: 'my-dataset-name',
    keyFilePath: 'path_to_account_service_credentials.json',
  });
await classifier.setupVertexClassifier();
await classifier.addToDataset("my-label-1", video_file1) //an File object of class video. The classifier will extract the respective frames and upload it to dataset bucket. 
await classifier.addToDataset("my-label-2", video_file2)

/* Trainig pipeline is divided in:
1- create a dataset in ai platform.
2- import data from the GCS storage bucket to dataset.
3- creates a training pipeline in ai platform with auto ML.
4- creates a endpoint for prediction
5- upload trained model to endpoint.
*/
await classifier.startTraining()
```