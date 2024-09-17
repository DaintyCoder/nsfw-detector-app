import React, { useState, useRef } from "react";
import cv from "@techstark/opencv-js";
import { Tensor, InferenceSession } from "onnxruntime-web";
import Loader from "./components/Loader";
import { detectImage } from "./utils/detect";
import { download } from "./utils/download";
import { LoadingState } from "./routes/types";
import "./styles/App.css";

interface Session {
  net: InferenceSession;
  nms: InferenceSession;
}

const App: React.FC = () => {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState<LoadingState | null>({ text: "Loading OpenCV.js", progress: null });
  const [image, setImage] = useState<string | null>(null);
  const [isNSFW, setIsNSFW] = useState<boolean>(false);
  const inputImage = useRef<HTMLInputElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Configs
  const modelName = "320n.onnx";
  const modelInputShape = [1, 3, 320, 320];
  const topk = 100;
  const iouThreshold = 0.45;
  const scoreThreshold = 0.25;

  // wait until opencv.js initialized
  cv["onRuntimeInitialized"] = async () => {
    const baseModelURL = "./model"; // Directly reference the model folder in the public directory
    console.log("Base Model URL:", baseModelURL);
    try {
      // create session
      const encodedModelName = encodeURIComponent(modelName);
      const modelURL = `${baseModelURL}/${encodedModelName}`;
      console.log("Model URL:", modelURL);
      const arrBufNet = await download(modelURL, ["Loading NudeDetector", setLoading]);
      console.log("Downloaded model size:", arrBufNet.byteLength);
      let yolov8;
      try {
        yolov8 = await InferenceSession.create(arrBufNet);
      } catch (error) {
        console.error("Error creating InferenceSession for yolov8:", error);
        setLoading({ text: "Failed to load model", progress: null });
        return;
      }
      const nmsModelURL = `${baseModelURL}/nms-yolov8.onnx`;
      console.log("NMS Model URL:", nmsModelURL);
      const arrBufNMS = await download(nmsModelURL, ["Loading NMS model", setLoading]);
      const nms = await InferenceSession.create(arrBufNMS);

      // warmup main model
      setLoading({ text: "Warming up model...", progress: null });
      const tensor = new Tensor("float32", new Float32Array(modelInputShape.reduce((a, b) => a * b)), modelInputShape);
      await yolov8.run({ images: tensor });
      setSession({ net: yolov8, nms: nms });
      setLoading(null); // Indicate that loading is done
    } catch (error) {
      console.error("Error initializing model:", error);
      setLoading({ text: "Failed to load model", progress: null });
    }
  };

  return (
    <div className="App">
      {loading && (
        <Loader>
          {loading.progress ? `${loading.text} - ${loading.progress}%` : loading.text}
        </Loader>
      )}
      <div className="header">
        <h1>nudenet: NSFW Detector in-browser</h1>
      </div>
      <div className="content">
        <img
          ref={imageRef}
          src="#"
          alt=""
          className={image ? "visible" : "hidden"}
          onLoad={async () => {
            if (session) {
              const nsfw = await detectImage(imageRef.current!, canvasRef.current!, session, topk, iouThreshold, scoreThreshold, modelInputShape);
              setIsNSFW(nsfw);
            }
          }}
        />
        <canvas id="canvas" width={modelInputShape[2]} height={modelInputShape[3]} ref={canvasRef} />
        {isNSFW && <p className="nsfw-warning">Warning: This image contains NSFW content.</p>}
      </div>
      <label htmlFor="fileInput" className="hidden">Upload Image</label>
      <input
        id="fileInput"
        type="file"
        ref={inputImage}
        accept="image/*"
        className="hidden"
        title="Upload Image"
        onChange={(e) => {
          if (image) {
            URL.revokeObjectURL(image);
            setImage(null);
            setIsNSFW(false);
          }
          const file = e.target.files?.[0];
          if (file) {
            const url = URL.createObjectURL(file); // create image url
            imageRef.current!.src = url; // set image source
            setImage(url);
          }
        }}
      />
      <div className="btn-container">
        <button onClick={() => { inputImage.current?.click(); }}>
          Open local image
        </button>
        {image && (
          <button onClick={() => {
            if (inputImage.current) {
              inputImage.current.value = "";
            }
            if (imageRef.current) {
              imageRef.current.src = "#";
            }
            URL.revokeObjectURL(image);
            setImage(null);
            setIsNSFW(false);
          }}>
            Close image
          </button>
        )}
      </div>
    </div>
  );
};

export default App;