if(typeof(require) === 'function'){

    const {
        deepTrain,
        getDeepModel,
        dsToDeepDS,
        FFN1D,
        RNNShared,
        LSTMCell
    } = require('./deepmodels')

}else{

    deepTrain = deepmodels.deepTrain;
    getDeepModel = deepmodels.getDeepModel;
    dsToDeepDS = deepmodels.dsToDeepDS;
    FFN1D = deepmodels.FFN1D;
    RNNShared = deepmodels.RNNShared;
    LSTMCell = deepmodels.LSTMCell;
}

// Start of module boilerplate, insures that code is useable both
// on server side and client side
(function(exports){

function getRandSequence(seq_len=1000, vocab_size=6, noise=0, dim=1){
    let seq = []
    for(let i = 0; i < seq_len;  i += 1){
        seq.push(new Array());
        for(let j = 0; j < dim; j++){
            seq[i].push(
                Math.floor(Math.random() * vocab_size) + 1 +
                noise * (Math.random() - 0.5)
            )
        }
    }
    return(seq)
}

function getPalindrome(vocab_size=10, halfSeqLen=0, noise=0, dim=1){
    if(halfSeqLen == 0){
        halfSeqLen = Math.round(Math.random() * (5 - 1)) + 1;
    }
    const first_half = getRandSequence(halfSeqLen, vocab_size, noise, dim);
    let i = first_half.length;
    let second_half = []
    while(i--){ second_half.push(first_half[i]) };
    const seq = first_half.concat(second_half)
    return(seq);
}

function getNonPalindrome(vocab_size=10, halfSeqLen=0, noise=0, dim=1){
    if(halfSeqLen == 0){
        halfSeqLen = Math.round(Math.random() * (5 - 1)) + 1;
    }
    const first_half = getRandSequence(halfSeqLen, vocab_size, noise, dim);
    let second_half = getRandSequence(halfSeqLen, vocab_size, noise, dim);

    let same = true;
    for(let i = 0; i < first_half.length; i++){
        for(let j = 0; j < first_half[i].length; j++){
            if(
                Math.round(second_half[second_half.length - (i + 1)][j]) !=
                Math.round(first_half[i][j])
            ){
                same = false;
            }
        }
    }
    if(same){
        ref_pos = Math.floor(Math.random() * halfSeqLen);
        change_pos = second_half.length - (ref_pos + 1);
        let feat = Math.floor(Math.random() * second_half[0].length);
        while(
            Math.round(second_half[change_pos][feat]) ==
            Math.round(first_half[ref_pos][feat])
        ){
             second_half[change_pos][feat] = Math.floor(
                 Math.random() * vocab_size
             ) + 1 + noise * (Math.random() - 0.5);
        }
    }
    const seq = first_half.concat(second_half)
    return(seq);
}

function getSequence(){
    let seq = []
    for(i = 0; i < 500;  i += 1){
        for(j = 0; j < 10;  j += 1){
            if(i % 2 == 0){
                seq.push([0])
            }else{
                seq.push([1000])
            }
        }
    }
    return(seq)
}

function arbitrarySeqLstmSynaptic(){
    // var synaptic = require('synaptic'); NODEjs-y...
    // var LSTM = new synaptic.Architect.LSTM(10,32,2);
    var LSTM = new synaptic.Architect.LSTM(10,32,2);
    return(LSTM);
}

function speedTest(n_repeat){
    let seq = getSequence();
    let LSTM = arbitrarySeqLstmSynaptic();
    let start = new Date().getTime(); // in millis
    for(i = 0; i <= n_repeat; i++){
        LSTM.activate(seq.slice(0, 10));
    }
    let end = new Date().getTime(); // in millis
    console.log(
        ''.concat(...[
            'time per predict: ', String(((end - start) / 1000) / n_repeat),
            's'
        ])
    );
    return({seq: seq, LSTM: LSTM});
}

function drawSequence(
    sequence, context, height, height_offset, color="#AB0F0F", divisions=5
){

    context.strokeStyle = color
    context.fillStyle = color

    let index_element_it = sequence.entries()
    for([idx, elt] of index_element_it){
        var [x, y] = [
            50 * (idx + 1),
            height_offset + ((height / divisions) * elt)
        ]
        if(idx > 0){
            context.lineTo(x, y);
            context.closePath()
            context.stroke()
        }
        context.beginPath()
        context.arc(x, y, 5, 0, 2 * Math.PI)
        context.closePath()
        context.fill()
        context.beginPath()
        context.moveTo(x, y)
    }

}

function getFFNBrain({hidden_size=[2]}){

    var net = new brain.NeuralNetwork({
        hidden_layers: hidden_size
    });
    return(net)
}

function getLstmBrain({hidden_size=[2]}){

    var net = new brain.recurrent.LSTMCell({
        hidden_layers: hidden_size
    });
    return(net)
}


function getFFNSynaptic({
    input_size=1, hidden_size=[2], output_size=2
}){

    const Layer = synaptic.Layer;
    const Network = synaptic.Network;
    const Trainer = synaptic.Trainer;

    const inputLayer = new Layer(input_size);
    let prev_layer = inputLayer;
    let hidden_layers = [];
    let last_layer;
    for(size of hidden_size){
        const hiddenLayer = new Layer(size);
        hidden_layers.push(hiddenLayer);
        prev_layer.project(hiddenLayer);
        last_layer = hiddenLayer;
    }

    const outputLayer = new Layer(output_size);

    last_layer.project(outputLayer);

    const net = new Network({
        input: inputLayer,
        hidden: hidden_layers,
        output: outputLayer
    });

    return(net)

}

function setColor(is_pal){
    if(is_pal){
        return("#11AE11")
    }
    else{
        return("#AB0F0F")
    };
}

function sleep(time){
  return new Promise((resolve) => setTimeout(resolve, time));
}

function reassort({dim=null, seq=null}){
    let seq_mdim = new Array();
    let step_count = Math.floor(seq.length / dim);
    console.log(step_count);
    console.log(seq);
    for(let step= 0; step < step_count; step++){
        seq_mdim.push(new Array());
        for(let feat = 0; feat < dim; feat++){
            seq_mdim[step].push(seq[step * dim + step]);
        }
    }
    return(seq_mdim);
}

function getPalindromeDataset({
    datasetSize=1000, flatten=true, halfSeqLen=1, vocab_size=2,
    noise=0, dim=1, outputLength=1
}){

    let trainSet = []
    for(let i = 0; i < datasetSize / 2; i++){
        let pal = getPalindrome(
            vocab_size=vocab_size, halfSeqLen=halfSeqLen, noise=noise, dim=dim
        );
        let non_pal = getNonPalindrome(
            vocab_size=vocab_size, halfSeqLen=halfSeqLen, noise=noise, dim=dim
        );
        if(flatten){
            pal = [].concat(...pal);
            non_pal = [].concat(...non_pal);
        }
        /*
        if(dim > 1){
            pal = reassort({seq: pal, dim: dim});
            non_pal = reassort({seq: non_pal, dim: dim});
        }
        */
        /*
            trainSet.push({input: pal, output: [0,1]});
            trainSet.push({input: non_pal, output: [1,0]});
        */
        let target = outputLength === 1 ? [1] : [0, 1];
        trainSet.push({input: pal, output: target});
        target = outputLength === 1 ? [0] : [1, 0];
        trainSet.push({input: non_pal, output: target});
    }
    return(trainSet);
}

function getMNISTDataset({
    datasetSize=700
}){
    const set = mnist.set(datasetSize, 20);
    return(set.training);
}

function getCheckerDataset({datasetSize=1000}){

    let trainSet = []
    for(let i = 0; i < datasetSize ; i++){
        x = Math.round(Math.random());
        y = Math.round(Math.random());
        is_same = (
            (x > 0.5) && (y > 0.5)
        ) || (
            (x <= 0.5) && (y <= 0.5)
        )
        trainSet.push({input: [x, y], output: [! is_same, is_same]});
    }
    return(trainSet);

}

async function learnModel({
    model_type='ffn_synaptic', hidden_size=[1],
    rate=0.2, iterations=100, error=0.005, train_set_size=1000,
    trainSet=null, input_size=1, output_size=1, momentum=0.9,
    batchSize=64, seqLength=null, optimizer=null
}){

    console.assert(typeof(rate) === 'number');
    console.assert(typeof(iterations) === 'number');
    console.assert(typeof(error) === 'number');
    console.assert(typeof(train_set_size) === 'number');
    console.assert(typeof(input_size) === 'number');
    console.assert(typeof(output_size) === 'number');
    console.assert(typeof(momentum) === 'number');
    console.assert(typeof(batchSize) === 'number');
    console.assert(typeof(seqLength) === 'number');

    let start = new Date().getTime(); // in millis

    const Architect = synaptic.Architect;
    const Layer = synaptic.Layer;
    const Trainer = synaptic.Trainer;

    let model;

    let getDeep = (x) => {
        let model = getDeepModel({
            modelType: x,
            nPredictions: output_size,
            hiddenSize: hidden_size[0],
            inputSize: input_size,
            seqLength: seqLength
        });
        return(model);
    }

    switch(model_type){
        case "lstm_synaptic":
            model = new Architect.LSTMCell(
                input_size, hidden_size, output_size
            );
            break;
        case "perceptron_synaptic":
            model = new Architect.Perceptron(
                input_size, hidden_size, output_size
            );
            break;
        case "ffn_synaptic":
            model = getFFNSynaptic({
                "input_size": input_size,
                "hidden_size": hidden_size,
                "output_size": output_size
            });
            break
        case 'ffn_brain':
            model = getFFNBrain({
                "hidden_size": hidden_size,
            });
            break;
        case 'lstm_brain':
            model = getLstmBrain({
                "hidden_size": hidden_size,
            });
            break;
        case 'lstmcell_deeplearn':
            model = getDeep('LSTM');
            break;
        case 'lstm_deeplearn':
            model = getDeep('RNNLSTM');
            break;
        case 'ffn_deeplearn':
            model = getDeep('FFN1D');
            break;
        default:
            console.log('Error no such model option: ', model_type);
    }

    if(model_type.match('.*_synaptic')){
        const trainOptions = {
            rate: rate,
            iterations: iterations,
            error: error,
            cost: Trainer.cost.CROSS_ENTROPY, // Trainer.cost.BINARY,
            log: true,
            crossValidate: null
        };

        const trainer = new Trainer(model);
        const trainResults = trainer.train(trainSet, trainOptions);

        console.log(trainSet);
        console.log(trainResults);
    }else if(model_type.match('.*_brain')){
        model.train(
            trainSet, {
              errorThresh: 0.005,  // error threshold to reach
              iterations: iterations,   // maximum training iterations
              log: true,           // console.log() progress periodically
              logPeriod: 10,       // number of iterations between logging
              learningRate: rate    // learning rate
            }
        );
    }else if(model_type.match('.*_deeplearn')){

        let dlDS = dsToDeepDS({
            ds: trainSet,
            dim: input_size,
            flatten: true,
            seqLen: seqLength,
            outputSize: output_size,
            hiddenSize: hidden_size[0],
            make2d: true
        });

        await deepTrain({
            model: model,
            printInfo: false,
            dsProviders: dlDS,
            batchSize: batchSize,
            learningRate: rate,
            momentum: momentum,
            iterations: iterations,
            optimizerType: optimizer
        });

    }

    let end = new Date().getTime(); // in millis
    console.log(
        ''.concat(...[
            'time for learn: ', String(((end - start) / 1000)),
            's'
        ])
    );


    return(model);
}

function testBatch({
    canvas, context, curr, max, ms, model, testSet=null, model_type=null
}){

    console.log(curr);
    console.log('testSet:', testSet);

    context.clearRect(0, 0, canvas.width, canvas.height);

    sub_height = canvas.height / testSet.length;
    var to_draw_it = testSet.entries();

    let pred;

    for([idx, {input: seq, output: output}] of to_draw_it){
        //seq = input;
        is_pal = Boolean(output[0]);
        drawSequence(
            seq, context, sub_height, sub_height * idx, setColor(is_pal)
        );


        if(model_type.match('.*_synaptic')){
            pred = model.activate(seq);
        }else if(model_type.match('.*_brain')){
            pred = model.run(seq);
        }

        console.log(is_pal, pred, seq);
    }

    if(model_type.match('.*_brain')){
        stats = model.test(testSet);
        console.log(testSet[0].output.length);
        console.log(stats);
    }

    curr = curr + 1;

    if(curr < max){
        console.log("+curr:", curr);
        console.log("+max:", max);
        setTimeout(
            () => testBatch({
            canvas: canvas, context: context, curr: curr, max: max, ms: ms,
            model: model, seq_len: 1, vocab_size: 2
            })
        );
        // sleep(ms).then(draw_batches(curr, max))
    }else{
        console.log("curr:", curr);
        // console.log("max:", max)
    }
}

function brainTrainInit(data, _options){
    const options = Object.assign({}, this.constructor.trainDefaults, _options);
    data = this.formatData(data);
    let iterations = options.iterations;
    let errorThresh = options.errorThresh;
    let log = options.log === true ? console.log : options.log;
    let logPeriod = options.logPeriod;
    let learningRate = _options.learningRate || this.learningRate || options.learningRate;
    let callback = options.callback;
    let callbackPeriod = options.callbackPeriod;

    if (!options.reinforce) {
      let sizes = [];
      let inputSize = data[0].input.length;
      let outputSize = data[0].output.length;
      let hiddenSizes = this.hiddenSizes;
      if (!hiddenSizes) {
        sizes.push(Math.max(3, Math.floor(inputSize / 2)));
      } else {
        hiddenSizes.forEach(size => {
          sizes.push(size);
        });
      }

      sizes.unshift(inputSize);
      sizes.push(outputSize);

      this.initialize(sizes);
    }

    return({
        data: data,
        iterations: iterations,
        errorThresh: errorThresh,
        log: log,
        logPeriod: logPeriod,
        learningRate: learningRate,
        callback: callback,
        callbackPeriod: callbackPeriod
    })
}

function brainIteration({
    data = null, i = 0, errorThresh = null, callback = null,
    callbackPeriod = null, logPeriod = null, learningRate = null,
    log = null
}){

    let sum = 0;
    for (let j = 0; j < data.length; j++) {
        let err = this.trainPattern(
            data[j].input, data[j].output, learningRate
        );
        sum += err;
    }
    let error = sum / data.length;

    if (log && (i % logPeriod === 0)) {
        log('iterations:', i, 'training error:', error);
    }
    if (callback && (i % callbackPeriod === 0)) {
        callback({ error: error, iterations: i });
    }

    return({
      error: error,
    });

}

var debug = {trainSet: null, testSet: null, model: null}

// END of export boiler plate
exports.getPalindromeDataset = getPalindromeDataset;
exports.drawSequence = drawSequence;
exports.getFFNBrain = getFFNBrain;
exports.getLstmBrain = getLstmBrain;
exports.learnModel = learnModel;
exports.getFFNSynaptic = getFFNSynaptic;
exports.speedTest = speedTest;
exports.testBatch = testBatch;
exports.debug = debug;
exports.brainIteration = brainIteration;
exports.brainTrainInit = brainTrainInit;
})(
    typeof exports === 'undefined'?  this['utils']={}: exports
);


/*
biblio:

http://caza.la/synaptic/#/
https://github.com/cazala/synaptic
https://tenso.rs/
https://deeplearnjs.org/
https://medium.freecodecamp.org/how-to-create-a-neural-network-in-javascript-in-only-30-lines-of-code-343dafc50d49
https://cs.stanford.edu/people/karpathy/convnetjs/
https://tutorialzine.com/2017/04/10-machine-learning-examples-in-javascript
https://github.com/BrainJS/brain.js
https://github.com/karpathy/convnetjs
https://github.com/deepforge-dev/deepforge
https://www.npmjs.com/package/tensorflow2
https://bower.io/
https://github.com/cazala/synaptic/blob/master/dist/synaptic.min.js
*/
