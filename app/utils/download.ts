import { LoadingState } from '../routes/types';

export const download = (
  url: string,
  logger: [string, React.Dispatch<React.SetStateAction<LoadingState | null>>] | null = null
): Promise<ArrayBuffer> => {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("GET", url, true);
    request.responseType = "arraybuffer";

    if (logger) {
      const [log, setState] = logger;
      request.onprogress = (e) => {
        const progress = (e.loaded / e.total) * 100;
        setState((prevState) => ({
          ...prevState,
          text: log,
          progress: parseFloat(progress.toFixed(2)),
        }));
      };
    }

    request.onload = function () {
      if (this.status >= 200 && this.status < 300) {
        resolve(request.response);
      } else {
        reject({
          status: this.status,
          statusText: request.statusText,
        });
      }
    };

    request.onerror = function () {
      reject({
        status: this.status,
        statusText: request.statusText,
      });
    };

    request.send();
  });
};