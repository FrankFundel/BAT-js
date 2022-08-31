const ort = require("onnxruntime-node");
const { ArgumentParser } = require("argparse");

const parser = new ArgumentParser({
  description: "Automatic bat call classification",
});
parser.add_argument("path", {
  help: "Path to directory where audio files are located (required)",
});
parser.add_argument("--threshold", {
  help: "Threshold for prediction (if -1 threshold will be determined automatically).",
  default: 0.5,
});
parser.add_argument("--expanded", {
  action: "store_true",
  help: "If files are already expanded 1:10.",
});
args = parser.parse_args();

console.log(args);

// read file
// (expand)
// highpass filter
// spectrogram
// noise filter
// sequencing (maybe torch.unfold)

const main = async () => {
  const session = await ort.InferenceSession.create("BAT.onnx");

  const inputs = {
    input: new ort.Tensor(
      "float32",
      new Float32Array(60 * 44 * 257),
      [1, 60, 44, 257]
    ),
  };
  const results = await session.run(inputs);
  console.log(results.output.data);
};
main();
