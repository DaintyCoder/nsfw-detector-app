import React from "react";
import "../styles/loader.css";

interface LoaderProps {
  children?: React.ReactNode;
}

const Loader: React.FC<LoaderProps> = (props) => {
  return (
    <div className="wrapper" {...props}>
      <div className="spinner"></div>
      <p>{props.children}</p>
    </div>
  );
};

export default Loader;