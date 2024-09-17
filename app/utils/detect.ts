import cv, { Mat } from "@techstark/opencv-js";
import { Tensor, InferenceSession } from "onnxruntime-web";
import { renderBoxes } from "./renderBox";

interface Session {
  net: InferenceSession;
  nms: InferenceSession;
}

interface PreprocessingResult {
  input: Mat;
  xRatio: number;
  yRatio: number;
}

interface Box {
  label: number;
  probability: number;
  bounding: [number, number, number, number];
}

const NSFW_LABELS = [
  "FEMALE_GENITALIA_EXPOSED",
  "BUTTOCKS_EXPOSED",
  "FEMALE_BREAST_EXPOSED",
  "MALE_GENITALIA_EXPOSED",
  "ANUS_EXPOSED",
  "ARMPITS_EXPOSED",
  "BELLY_EXPOSED",
  "MALE_BREAST_EXPOSED"
];

const LABELS = [
  "FEMALE_GENITALIA_COVERED",
  "FACE_FEMALE",
  "BUTTOCKS_EXPOSED",
  "FEMALE_BREAST_EXPOSED",
  "FEMALE_GENITALIA_EXPOSED",
  "MALE_BREAST_EXPOSED",
  "ANUS_EXPOSED",
  "FEET_EXPOSED",
  "BELLY_COVERED",
  "FEET_COVERED",
  "ARMPITS_COVERED",
  "ARMPITS_EXPOSED",
  "FACE_MALE",
  "BELLY_EXPOSED",
  "MALE_GENITALIA_EXPOSED",
  "ANUS_COVERED",
  "FEMALE_BREAST_COVERED",
  "BUTTOCKS_COVERED"
];

/**
 * Detect Image
 * @param {HTMLImageElement} image Image to detect
 * @param {HTMLCanvasElement} canvas canvas to draw boxes
 * @param {Session} session YOLOv8 onnxruntime session
 * @param {number} topk Integer representing the maximum number of boxes to be selected per class
 * @param {number} iouThreshold Float representing the threshold for deciding whether boxes overlap too much with respect to IOU
 * @param {number} scoreThreshold Float representing the threshold for deciding when to remove boxes based on score
 * @param {number[]} inputShape model input shape. Normally in YOLO model [batch, channels, width, height]
 * @return {boolean} Returns true if the image is NSFW, otherwise false
 */
export const detectImage = async (
  image: HTMLImageElement,
  canvas: HTMLCanvasElement,
  session: Session,
  topk: number,
  iouThreshold: number,
  scoreThreshold: number,
  inputShape: number[]
): Promise<boolean> => {
  const [modelWidth, modelHeight] = inputShape.slice(2);
  const { input, xRatio, yRatio } = preprocessing(image, modelWidth, modelHeight);
  const tensor = new Tensor("float32", input.data32F as Float32Array, inputShape); // to ort.Tensor
  const config = new Tensor(
    "float32",
    new Float32Array([
      topk, // topk per class
      iouThreshold, // iou threshold
      scoreThreshold, // score threshold
    ])
  ); // nms config tensor
  const { output0 } = await session.net.run({ images: tensor }); // run session and get output layer
  const { selected } = await session.nms.run({ detection: output0, config: config }); // perform nms and filter boxes

  const boxes: Box[] = [];
  let isNSFW = false;

  // Ensure `selected.data` is properly cast as Float32Array
  const selectedData = selected.data as Float32Array;

  // Loop through output
  for (let idx = 0; idx < selected.dims[1]; idx++) {
    // Correctly slice the data for each box, converting it to an array if needed
    const data = Array.from(selectedData.slice(idx * selected.dims[2], (idx + 1) * selected.dims[2]));
    const box = data.slice(0, 4) as number[];
    const scores = data.slice(4) as number[];
    const score = Math.max(...scores);
    const label = scores.indexOf(score);
    const [x, y, w, h] = [
      (box[0] - 0.5 * box[2]) * xRatio, // upscale left
      (box[1] - 0.5 * box[3]) * yRatio, // upscale top
      box[2] * xRatio, // upscale width
      box[3] * yRatio, // upscale height
    ]; // upscale box

    boxes.push({ label: label, probability: score, bounding: [x, y, w, h] });

    // Check if the label is in the NSFW list and the probability is above 60%
    if (NSFW_LABELS.includes(LABELS[label]) && score > 0.6) {
      isNSFW = true;
    }
  }

  console.log(boxes); // log boxes
  renderBoxes(canvas, boxes); // Draw boxes
  input.delete(); // delete unused Mat

  return isNSFW;
};

/**
 * Preprocessing image
 * @param {HTMLImageElement} source image source
 * @param {number} modelWidth model input width
 * @param {number} modelHeight model input height
 * @return preprocessed image and configs
 */
const preprocessing = (source: HTMLImageElement, modelWidth: number, modelHeight: number): PreprocessingResult => {
  const mat = cv.imread(source); // read from img tag
  const matC3 = new cv.Mat(mat.rows, mat.cols, cv.CV_8UC3); // new image matrix
  cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR); // RGBA to BGR

  // padding image to [n x n] dim
  const maxSize = Math.max(matC3.rows, matC3.cols); // get max size from width and height
  const xPad = maxSize - matC3.cols; // set xPadding
  const xRatio = maxSize / matC3.cols; // set xRatio
  const yPad = maxSize - matC3.rows; // set yPadding
  const yRatio = maxSize / matC3.rows; // set yRatio
  const matPad = new cv.Mat(); // new mat for padded image
  cv.copyMakeBorder(matC3, matPad, 0, yPad, 0, xPad, cv.BORDER_CONSTANT); // padding black

  const input = cv.blobFromImage(
    matPad,
    1 / 255.0, // normalize
    new cv.Size(modelWidth, modelHeight), // resize to model input size
    new cv.Scalar(0, 0, 0),
    true, // swapRB
    false // crop
  ); // preprocessing image matrix

  // release mat opencv mat.delete();
  matC3.delete();
  matPad.delete();

  return { input, xRatio, yRatio };
};