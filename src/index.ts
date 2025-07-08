import { Bucket, Storage, File } from '@google-cloud/storage';
import {
  DatasetServiceClient,
  PipelineServiceClient,
  EndpointServiceClient,
} from '@google-cloud/aiplatform';
import os from 'os';
import fs from 'fs';
import * as path from 'path';
import ffmpeg from 'fluent-ffmpeg';
import ffmpegInstaller from '@ffmpeg-installer/ffmpeg';

class VertexClassifier {
  private readonly datasetFilePath = `dataset.csv`;
  private readonly datasetPath = 'dataset/';
  private readonly projectId: string;
  private readonly location: string;
  private readonly vertexDatasetName: string;
  private readonly datasetBucketName: string;
  private readonly keyFilePath;
  private datasetBucket: Bucket;

  constructor({
    projectId,
    location,
    datasetBucketName,
    vertexDatasetName,
    keyFilePath,
  }: {
    projectId: string;
    location: string;
    datasetBucketName: string;
    vertexDatasetName: string;
    keyFilePath: string;
  }) {
    this.projectId = projectId;
    this.location = location;
    this.vertexDatasetName = vertexDatasetName;
    this.datasetBucketName = datasetBucketName;
    this.keyFilePath = keyFilePath;

    this.datasetBucket = new Storage({
      projectId: projectId,
      keyFilename: keyFilePath,
    }).bucket(datasetBucketName);
  }

  /**
   * Check if the dataset bucket contains a dataset list file, and create it if it doesn't. 
   */
  public async setupVertexClassifier() {
    const file = this.datasetBucket.file(this.datasetFilePath);
    const [exists] = await file.exists();
    if (!exists) {
      // Create the dataset file if it doesn't exist
      await file.save('GCS_FILE_PATH,LABEL\n', {
        contentType: 'text/csv',
      });
      console.log('Dataset file created:', this.datasetFilePath);
    }
  }

  /**
   * 
   * @returns the dataset length
   */
  async getDatasetLength(): Promise<number> {
    const localPath = os.tmpdir() + '/dataset.csv';
    await this.datasetBucket
      .file(this.datasetFilePath)
      .download({ destination: localPath });
    const datasetCSV = fs.readFileSync(localPath, 'utf8');
    const classes: String[] = [];
    datasetCSV
      .replace('GCS_FILE_PATH,LABEL\n', '')
      .split('\n')
      .forEach((line) => {
        const className = line.split(',')[1];
        if (className && !classes.includes(className)) {
          classes.push(className);
        }
      });
    return classes.length;
  }

  /**
   * 
   * @param id - class label
   * @param trainVideo - video file of the object from this class
   * 
   * add video frames to dataset bucket and updates dataset list file
   */
  async addToDataset(id: string, trainVideo: File): Promise<void> {
    const localPath = os.tmpdir() + '/dataset.csv';

    const file = this.datasetBucket.file(this.datasetFilePath);

    await file.download({ destination: localPath });
    let content = fs.readFileSync(localPath, 'utf8');

    const classPath = `${this.datasetPath}${id}/`;
    let [files] = await this.datasetBucket.getFiles({
      prefix: classPath,
      maxResults: 1,
    });
    if (files.length == 0) {
      const inputLocalVideoPath = path.join(os.tmpdir(), `video_${id}.mp4`);
      await trainVideo.download({ destination: inputLocalVideoPath });
      const outLocalPath = path.join(os.tmpdir(), `frames_${id}`);
      const frameFiles = await this.extractFrames(
        inputLocalVideoPath,
        outLocalPath,
        10,
      );

      for (const frameFile of frameFiles) {
        const localFramePath = path.join(outLocalPath, frameFile);

        console.log(
          `Uploading ${localFramePath} to gs://${this.datasetBucketName}/${classPath}`,
        );
        await this.datasetBucket.upload(localFramePath, {
          destination: `${classPath}${frameFile}`,
          metadata: {
            contentType: 'image/png',
          },
        });
      }
      //delete temp files
      console.log('Cleaning up temporary files...');
      if (fs.existsSync(inputLocalVideoPath)) {
        fs.unlinkSync(inputLocalVideoPath);
        console.log(`Deleted ${inputLocalVideoPath}`);
      }
      if (fs.existsSync(outLocalPath)) {
        fs.rmSync(outLocalPath, { recursive: true, force: true });
        console.log(`Deleted ${outLocalPath}`);
      }
    }

    const [currentFiles] = await this.datasetBucket.getFiles({
      prefix: classPath,
    });

    if (!content.endsWith('\n')) {
      content += '\n';
    }
    for (const file of currentFiles) {
      content += `gs://${this.datasetBucketName}/${file.name},${id}\n`;
    }

    fs.writeFileSync(localPath, content, 'utf8');

    await this.datasetBucket.upload(localPath, {
      destination: this.datasetFilePath,
      metadata: {
        contentType: 'text/csv',
      },
    });
  }

  /**
   * 
   * @param id - class label
   * 
   * remove images from dataset bucket
   */
  async removeFromDataset(id: string): Promise<void> {
    const classPath = `${this.datasetPath}${id}/`;
    await this.datasetBucket.deleteFiles({
      prefix: classPath,
    });
  }

  /**
   * 
   * @param id - class label
   * 
   * remove images from dataset list file
   */
  async removeFromDatasetFile(id: string): Promise<void> {
    const localPath = os.tmpdir() + '/dataset.csv';

    const file = this.datasetBucket.file(this.datasetFilePath);

    await file.download({ destination: localPath });
    let content = fs.readFileSync(localPath, 'utf8');

    const regex = new RegExp(`^.*?,${id}\\r?\\n`, 'gm');
    content = content.replace(regex, '');
    fs.writeFileSync(localPath, content, 'utf8');

    await this.datasetBucket.upload(localPath, {
      destination: this.datasetFilePath,
      metadata: {
        contentType: 'text/csv',
      },
    });
  }

  /**
   * starts training pipeline:
   * 
   * 1- create a dataset in ai platform. 
   * 
   * 2- import data from the GCS storage bucket to dataset.
   * 
   * 3- creates a training pipeline in ai platform with auto ML.
   * 
   * 4- creates a endpoint for prediction
   * 
   * 5- upload trained model to endpoint.
   * 
  */
  async startTraining(): Promise<void> {
    const apiEndpoint = `${this.location}-aiplatform.googleapis.com`;
    const clientOptions = {
      projectId: this.projectId,
      apiEndpoint,
      keyFilename: this.keyFilePath,
    };
    const datasetClient = new DatasetServiceClient(clientOptions);
    const pipelineClient = new PipelineServiceClient(clientOptions);
    const endpointClient = new EndpointServiceClient(clientOptions);

    const parent = `projects/${this.projectId}/locations/${this.location}`;
    let datasetFullName: string;

    try {
      // Delete last dataset (optional)
      const [datasets] = await datasetClient.listDatasets({ parent });
      if (datasets.length > 0) {
        await datasetClient.deleteDataset({ name: datasets[0].name! });
      }

      console.log('Creating dataset...');
      const [operation] = await datasetClient.createDataset({
        parent,
        dataset: {
          displayName: this.vertexDatasetName,
          metadataSchemaUri:
            'gs://google-cloud-aiplatform/schema/dataset/metadata/image_1.0.0.yaml',
        },
      });

      const [datasetResponse] = await operation.promise();
      datasetFullName = datasetResponse.name!;
      console.log('✅ Dataset created:', datasetFullName);
    } catch (e) {
      console.error('❌ Failed to create dataset:', e);
      return;
    }

    try {
      console.log('Importing data...');
      const [importOperation] = await datasetClient.importData({
        name: datasetFullName,
        importConfigs: [
          {
            gcsSource: {
              uris: [`gs://${this.datasetBucket.name}/${this.datasetFilePath}`],
            },
            importSchemaUri:
              'gs://google-cloud-aiplatform/schema/dataset/ioformat/image_classification_single_label_io_format_1.0.0.yaml',
          },
        ],
      });
      await importOperation.promise();
      console.log('✅ Data imported into dataset.');
    } catch (e) {
      console.error('❌ Failed to import data:', e);
      await datasetClient.deleteDataset({ name: datasetFullName });
      return;
    }

    let modelName: string;
    let modelId: string;
    try {
      console.log('Creating training pipeline...');
      const datasetId = datasetFullName.split('/').pop();

      const [trainingPipeline] = await pipelineClient.createTrainingPipeline({
        parent,
        trainingPipeline: {
          displayName: 'image-classification-model',
          trainingTaskDefinition:
            'gs://google-cloud-aiplatform/schema/trainingjob/definition/automl_image_classification_1.0.0.yaml',
          inputDataConfig: {
            datasetId,
          },
          modelToUpload: {
            displayName: 'image-classification-model',
          },
          trainingTaskInputs: {
            structValue: {
              fields: {
                //budgetMilliNodeHours: { numberValue: 8000 },
                modelType: { stringValue: 'CLOUD' },
              },
            },
          },
        },
      });

      const pipelineName = trainingPipeline.name!;
      console.log('⏳ Waiting for training pipeline to complete...');

      await this.waitForPipelineCompletion(pipelineClient, pipelineName);

      modelId = trainingPipeline.modelId!;
      modelName = trainingPipeline.modelToUpload!.displayName!;
      console.log('✅ Model trained:', modelName);
    } catch (e) {
      console.error('❌ Failed to train model:', e);
      await datasetClient.deleteDataset({ name: datasetFullName });
      return;
    }

    let endpointName: string;
    try {
      console.log('Creating endpoint...');
      const [endpointOp] = await endpointClient.createEndpoint({
        parent,
        endpoint: {
          displayName: 'image-endpoint',
        },
      });

      const [endpointResponse] = await endpointOp.promise();
      endpointName = endpointResponse.name!;
      console.log('✅ Endpoint created:', endpointName);
    } catch (e) {
      console.error('❌ Failed to create endpoint:', e);
      await datasetClient.deleteDataset({ name: datasetFullName });
      return;
    }

    try {
      console.log('Deploying model to endpoint...');
      const [deployOp] = await endpointClient.deployModel({
        endpoint: endpointName,
        deployedModel: {
          model: modelId,
          displayName: modelName,
        },
      });

      await deployOp.promise();
      console.log('✅ Model deployed to endpoint.');
    } catch (e) {
      console.error('❌ Failed to deploy model:', e);
      await endpointClient.deleteEndpoint({ name: endpointName });
      await datasetClient.deleteDataset({ name: datasetFullName });
      return;
    }
  }

  private async waitForPipelineCompletion(
    pipelineClient: PipelineServiceClient,
    pipelineName: string,
    interval = 60000,
  ): Promise<void> {
    while (true) {
      const [pipeline] = await pipelineClient.getTrainingPipeline({
        name: pipelineName,
      });
      const state = pipeline.state;

      console.log(`Training pipeline state: ${state}`);

      if (state === 'PIPELINE_STATE_SUCCEEDED') {
        console.log('✅ Training completed successfully.');
        break;
      } else if (state === 'PIPELINE_STATE_FAILED') {
        throw new Error('❌ Training failed.');
      } else if (state === 'PIPELINE_STATE_CANCELLED') {
        throw new Error('❌ Training was cancelled.');
      }

      // Wait before checking again
      await new Promise((res) => setTimeout(res, interval));
    }
  }

  private async extractFrames(
    inputLocalVideoPath: string,
    outLocalPath: string,
    framesToSkip: number,
  ): Promise<string[]> {
    ffmpeg.setFfmpegPath(ffmpegInstaller.path);

    try {
      fs.mkdirSync(outLocalPath, { recursive: true });

      console.log(
        `Extracting frames to ${outLocalPath}, skipping ${framesToSkip} frames between each.`,
      );
      await new Promise<void>((resolve, reject) => {
        ffmpeg(inputLocalVideoPath)
          .on('filenames', (filenames: string[]) => {
            console.log('Generated filenames locally: ', filenames);
          })
          .on('end', () => {
            console.log('Frame extraction finished.');
            resolve();
          })
          .on('error', (err: Error) => {
            console.error('Error during frame extraction:', err);
            reject(err);
          })
          .addOptions([
            `-vf`,
            `select=not(mod(n\\,${framesToSkip}))`,
            `-vsync`,
            `vfr`,
          ])
          .output(`${outLocalPath}/frame_%03d.png`)
          .run();
      });

      const extractedFrameFiles = fs
        .readdirSync(outLocalPath)
        .filter((file) => file.endsWith('.png'));

      console.log('All frames extracted successfully.');

      return extractedFrameFiles;
    } catch (error) {
      console.error('Error in extractFrames:', error);
      throw error;
    }
  }
}

export default VertexClassifier;
