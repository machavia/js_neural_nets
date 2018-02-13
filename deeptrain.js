if(typeof(require) === 'function'){

    const {
        ENV,
        Array1D,
        Array2D,
        Array3D,
        Scalar,
        Session,
        Tensor,
        util,
        NDArray,
        NDArrayMath,
        InCPUMemoryShuffledInputProviderBuilder,
        InGPUMemoryShuffledInputProviderBuilder
    } = require('./deeplearn')

}else{
    
    doBatch = deep_batch_train.doBatch;
    doBatchFromScratch = deep_batch_train.doBatchFromScratch;
    setOptimizerParams = deep_batch_train.setOptimizerParams;
    getOptimizerParams = deep_batch_train.getOptimizerParams;

    Feeder = deepmodels.Feeder;
    prepareFeed = deepmodels.prepareFeed;
    getOptimizer = deepmodels.getOptimizer;
    getDeepModel = deepmodels.getDeepModel;
    Feeder = deepmodels.Feeder;

    ENV = deeplearn.ENV;
    Array1D = deeplearn.Array1D;
    Array2D = deeplearn.Array2D;
    Array3D = deeplearn.Array3D;
    Scalar = deeplearn.Scalar;
    Session = deeplearn.Session;
    Tensor = deeplearn.Tensor;
    util = deeplearn.util;
    NDArray = deeplearn.NDArray;
    NDArrayMath = deeplearn.NDArrayMath;
    InCPUMemoryShuffledInputProviderBuilder = deeplearn.InCPUMemoryShuffledInputProviderBuilder;
    InGPUMemoryShuffledInputProviderBuilder = deeplearn.InGPUMemoryShuffledInputProviderBuilder;

}

// Start of module boilerplate, insures that code is useable both
// on server side and client side
(function(exports){

async function deepTrain({
    model=null,
    modelParams=null,
    printInfo=false,
    dsProviders=null,
    dsParameters=null,
    batchSize=64,
    learningRate=0.1,
    momentum=0.9,
    iterations=100,
    optimizerType='momemtum',
    optimizerByBatch=false,
    modelByBatch=false
}){

    // Houston we have a memory leak !
    // Only GPU ?

    console.assert(typeof(batchSize) === 'number');
    console.assert(typeof(learningRate) === 'number');
    console.assert(typeof(momentum) === 'number');
    console.assert(typeof(iterations) === 'number');


    // await firstLearn();
    let [
        feeder, graph, session, xFeed, lFeed, feedEntries, x_check, l_check,
        optimizer
    ] = [null, null, null, null, null, null, null, null, null]

    let [ds, xProvider, lProvider] = [
        null, null, null, null, null
    ]


    if (model === null){
        deeplearn.ENV.setMath(new deeplearn.NDArrayMath('cpu', true))
        math = ENV.math;
        // math.enableDebugMode();
    }

    if (modelByBatch){
        dsParameters.batchSize = batchSize;
        feeder = new Feeder(dsParameters);
    }else{
        [ds, xProvider, , , lProvider] = dsProviders;
    }

    // inject session with our modified version of train
    // maybe using bind to add the session context to the function ?
    // Ok rather than craping our pants here we will just modify the original
    // code...

    // else{ math = model.math; }
    // math.enableDebugMode();

    // weights =
    await math.scope(async () => {

        if(! modelByBatch){
            modelParams.math = math;
            model = getDeepModel(modelParams);
        }
        if(! (modelByBatch && optimizerByBatch)){
            feedEntries = prepareFeed(model, xProvider, lProvider);
        }
        if(! optimizerByBatch ){
            // Create an optimizer outside the loop
            // NOTE model is already provided if we are here
            optimizer = getOptimizer(optimizerType, learningRate, momentum);}

        let weightInit = null;
        let optimizerParams = null;

        for (let iter = 0; iter < iterations; iter++) {

            if(modelByBatch && optimizerByBatch){
                feedEntries = null;
                optimizer = null;
                model = null;
                modelParams["init_weights"] = weightInit;
            }

            let args = {};
            let batchFunc = null;
            if (modelByBatch && optimizerByBatch){
                args = {
                    batchSize: batchSize,
                    feedEntries: feedEntries,
                    iter: iter,
                    model: model,
                    optimizer: optimizer,
                    session: session
                };
                batchFunc = doBatchFromScratch;
            }
            else{
                args = {
                    batchSize: batchSize,
                    feeder: feeder,
                    iter: iter,
                    learningRate: learningRate,
                    model: model,
                    modelParams: modelParams,
                    momentum: momentum,
                    optimizerType: optimizerType,
                    optimizerParams: optimizerParams
                };
                batchFunc = doBatch;
            }

            ret = await batchFunc(args);
            [feedEntries, optimizer, model] = ret;

            if(modelByBatch && optimizerByBatch){
                weightInit = model.getWeightsValues();
                let data = model.math.backend.data;
                optimizerParams = getOptimizerParams(optimizer, data);
            }

            /*
            debug.feedEntries = feedEntries;
            debug.optimizer = optimizer;
            debug.model = model;
            debug.weights = model.getWeightsValues();
            debug.optimizerParams = optimizerParams;
            */

        }

    })

}

var debug = {}

// END of export boiler plate
exports.deepTrain = deepTrain;
exports.debug = debug;
exports.deep_batch_train = deep_batch_train;
})(
    typeof exports === 'undefined'?  this['deeptrain']={}: exports
);
