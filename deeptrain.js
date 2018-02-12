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

    // require('./node_modules/deeplearn/dist/deeplearn')

    // require('./node_modules/deeplearn/dist/deeplearn');

}else{
    
    doBatch = deep_batch_train.doBatch;
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
    InCPUMemoryShuffledInputProviderBuilderExtended = deeplearn.InCPUMemoryShuffledInputProviderBuilderExtended;
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

    if (modelByBatch){
        dsParameters.batchSize = batchSize;
        feeder = new Feeder(dsParameters);
    }else{
        [ds, xProvider, , , lProvider] = dsProviders;
    }
    // !!! REMOVE
    // [ds, xProvider, hProvider, cProvider, lProvider] = dsProviders;

    debug.ds = dsParameters;
    debug.xProvider = xProvider;
    debug.lProvider = lProvider;
    debug.model = model;
    debug.ENV = ENV;

    // inject session with our modified version of train
    // maybe using bind to add the session context to the function ?
    // Ok rather than craping our pants here we will just modify the original
    // code...

    if (model === null){
        deeplearn.ENV.setMath(new deeplearn.NDArrayMath('cpu', false))
        math = ENV.math;
    } // else{ math = model.math; }
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

        for (let iter = 0; iter < iterations; iter++) {

            if(modelByBatch && optimizerByBatch){
                feedEntries = null;
                optimizer = null;
                model = null;
            }

            let args = {
                batchSize: batchSize,
                debug: debug,
                ds: ds,
                feedEntries: feedEntries,
                feeder: feeder,
                graph: graph,
                iter: iter,
                lProvider: lProvider,
                learningRate: learningRate,
                math: math,
                model: model,
                modelByBatch: modelByBatch,
                modelParams: modelParams,
                momentum: momentum,
                optimizer: optimizer,
                optimizerByBatch: optimizerByBatch,
                optimizerType: optimizerType,
                printInfo: printInfo,
                session: session,
                xProvider: xProvider
            }

            /*
            optimizer
                accumulatedSquaredGradients 

            deeptrain.debug.optimizer.accumulatedSquaredGradients.dict
            // math deeptrain.debug.model.math.backend.data.dict[3]
            */

            /* ------>
            optimizer = deeptrain.debug.optimizer;
            gradientsDict = deeptrain.debug.optimizer.
                accumulatedSquaredGradients.dict;
            data = deeptrain.debug.model.math.backend.data

            for(gradientsVar of Object.values(gradientsDict)){
                console.log(gradientsVar);
                let gradientsVal = data[gradientsVar.dataId];
                console.log(gradientsVal)
                isSameSize = gradientsVar.size == gradientsVal.length;
                if(! isSameSize){
                    console.log(
                        "not the same size",
                        gradientsVar.size,
                        gradientsVal.length
                    )
                    break;
                }
            }
            */


            ret = await doBatch(args);
            [feedEntries, optimizer, model] = ret;

            debug.feedEntries = feedEntries;
            debug.optimizer = optimizer;
            debug.model = model;
            debug.weights = model.getWeightsValues();

            // Set proxy variables
            gradientsDict = optimizer.
                accumulatedSquaredGradients.dict;
            data = model.math.backend.data

            // get Square Gradients A and variables values with Names
            /*
            let vIds = [];
            console.log("====================================================")
            for(node of optimizer.variableNodes){
                console.log(node.name)
                console.log(node)
                let sqGradForNode = gradientsDict[node.id]
                console.log("==> SqGradNode", sqGradForNode);
                console.log("==> sqGradValue", data[sqGradForNode.dataId]);
                console.log("==> Value", data[node.data.dataId]);
            }
            */


            // get Gradient Values A (no names)
            /*
            for(
                [gradientsName, gradientsVar] of Object.entries(gradientsDict)
            ){
                console.log(gradientsName);
                console.log(gradientsVar);
                let gradientsVal = data[gradientsVar.dataId];
                console.log(gradientsVal)
                isSameSize = gradientsVar.size == gradientsVal.length;
                if(! isSameSize){
                    console.log(
                        "not the same size",
                        gradientsVar.size,
                        gradientsVal.length
                    )
                    break;
                }
            }
            */

            /*
            if(modelByBatch && optimizerByBatch){

                console.log("ABOOT")

                wk = new Worker('./deep_batch_train.js');

                console.log("BEN'D'LA MERDE");
                args.deepmodels = deepmodels;
                console.log("BEN'CON");
                ret = wk.postMessage(args);
                console.log("BEN'ZUT");

                console.log(ret);

                wk.terminate();

                console.log("A-BOOT")

            }
            else{
                await doBatch(args)
            }
            */

        }

        // return(weights);
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


/*
deeplearn.ENV.getBackend("cpu")
deeplearn.NDArrayMathCPU

// Everything ever tracked is here:
debugDeep.model.math.backendEngine.activeScope.track.forEach(node => {
    if(! node.isDisposed){
        console.log(node.dataId)
        node.dispose()
    }
})

*/
