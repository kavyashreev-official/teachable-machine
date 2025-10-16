let net;
const classifier = knnClassifier.create();
const webcamElement = document.getElementById('webcam');

async function setupWebcam() {
  return new Promise((resolve, reject) => {
    const navigatorAny = navigator;
    navigator.getUserMedia = navigator.getUserMedia || navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia;
    if (navigator.getUserMedia) {
      navigator.getUserMedia(
        { video: true },
        stream => {
          webcamElement.srcObject = stream;
          webcamElement.addEventListener('loadeddata', () => resolve(), false);
        },
        error => reject()
      );
    } else {
      reject();
    }
  });
}

async function app() {
  console.log('Loading mobilenet..');
  net = await mobilenet.load();
  console.log('Successfully loaded model');

  await setupWebcam();
}

app();

function addExample(classId) {
  const activation = net.infer(webcamElement, true);
  classifier.addExample(activation, classId);
}

async function predict() {
  if (classifier.getNumClasses() > 0) {
    const activation = net.infer(webcamElement, true);
    const result = await classifier.predictClass(activation);
    document.getElementById('prediction').innerText =
      `Prediction: Class ${result.label} (Confidence: ${result.confidences[result.label].toFixed(2)})`;
  }
}

