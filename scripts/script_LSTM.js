const {
    getPalindromeDataset,
    getFFNBrain,
    getLstmBrain,
    brainIteration,
    brainTrainInit
} = require("./utils");

brain = require("./bower_components/brain.js");

const cout = console.log;

let ds;

ds = getPalindromeDataset({
    datasetSize: 100,
    dim: 4,
    flatten: false,
    seq_len: 5
});

brain.recurrent.RNN.defaults.setupData(ds);

brain.recurrent.RNN.defaults.setupData(ds)[0].length;
// 13

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
let model;

model = getLstmBrain({hiddenSize: [32]});

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
      learningRate: 0.001    // learning rate
    }
);

model.model.equations

let predictions;

predictions = new Array();
for(let i = 0; i < ds.length; i++){
    predictions.push(model.run(ds[i].input));
}

/*
model.toJSON()
model.fromJSON()
*/
