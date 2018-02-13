const {
    getPalindromeDataset,
    getFFNBrain,
    getLstmBrain,
    brainIteration,
    brainTrainInit
} = require("./utils");

brain = require("./bower_components/brain.js");

const cout = console.log;

let ds = getPalindromeDataset({
    datasetSize: 1000,
    dim: 1,
    flatten: true,
    seq_len: 5
});

let trainOptions = {
    errorThresh: 0.005,  // error threshold to reach
    iterations: 1,   // maximum training iterations
    log: true,           // console.log() progress periodically
    logPeriod: 10,       // number of iterations between logging
    learningRate: 0.2    // learning rate
}

let prev_vars;
let count;
let err;

let model = getLstmBrain({hiddenSize: [6]});

model = getFFNBrain({hiddenSize: [16]});
model.brainTrainInit = brainTrainInit;
model.brainIteration = brainIteration;

cout(
"===========================================================================",
"\nMinibatch training",
"\n==========================================================================="
);

count = 0;
prev_vars = null;
while(count < 100){
    brainTrainOptions = model.brainTrainInit(ds, trainOptions);
    brainTrainOptions.i = count;
    if(prev_vars !== null){
        model.biases = prev_vars.biases;
        model.weights = prev_vars.weights;
        model.momentum = prev_vars.momentum;
    }
    err = model.brainIteration(brainTrainOptions).error;
    prev_vars = [];
    prev_vars.biases = model.biases;
    prev_vars.weights = model.weights;
    prev_vars.momentum = model.momentum;
    count++;
}

cout(
"===========================================================================",
"\nRegular training",
"\n==========================================================================="
);

model.train(
    ds, {
      errorThresh: 0.005,  // error threshold to reach
      iterations: 100,   // maximum training iterations
      log: true,           // console.log() progress periodically
      logPeriod: 10,       // number of iterations between logging
      learningRate: 0.2    // learning rate
    }
);
