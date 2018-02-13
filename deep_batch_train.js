function onMessage(e) {
    console.log("A BUTT");
    postMessage(e.data);
}

/*
if(typeof(require) === 'function'){
    // TODO nodejs compat
}else{
 
    getDeepModel = deepmodels.getDeepModel;
    prepareFeed = deepmodels.prepareFeed;
    getOptimizer = deepmodels.getOptimizer;

}
*/


// Start of module boilerplate, insures that code is useable both
// on server side and client side
(function(exports){

function getOptimizerParams(optimizer, data){

    // Set proxy variables
    gradientsDict = optimizer.accumulatedSquaredGradients.dict;

    const nameToParamValue = {};

    for(node of optimizer.variableNodes){
        let sqGradForNode = gradientsDict[node.id];
        nameToParamValue[node.name] = data[sqGradForNode.dataId];
    }

    return(nameToParamValue);
}

function setOptimizerParams(optimizer, nameToParamValue, data){

    // Set proxy variables
    gradientsDict = optimizer.accumulatedSquaredGradients.dict;

    for(node of optimizer.variableNodes){
        let sqGradForNode = gradientsDict[node.id];
        data[sqGradForNode.dataId] = nameToParamValue[node.name];
    }

}


async function doBatch({
    batchSize,
    debug,
    ds,
    feedEntries,
    feeder,
    graph,
    iter,
    lProvider,
    learningRate,
    math,
    model,
    modelByBatch,
    modelParams,
    momentum,
    optimizer,
    optimizerByBatch,
    optimizerType,
    printInfo,
    session,
    xProvider,
    optimizerParams=null
}){

    // we can chose to keep the same model / optimizer accross batches
    // or
    // we can chose to have a new model / optimizer for each batch
    if(modelByBatch){

        model = getDeepModel(modelParams);

        console.log("init_weights:", modelParams.init_weights)

        feeder.math = model.math;
        [xProvider, lProvider] = feeder.next();
    }

    // when setting optimizer_by_batch = false with
    // model_by_btach = false ==> error, optimizer by batch probably
    // keeps old refs
    if (optimizerByBatch){
        optimizer = getOptimizer(optimizerType, learningRate, momentum);
    }

    // console.log(model, optimizer);
    if (modelByBatch){
        feedEntries = prepareFeed(model, xProvider, lProvider);}

    // Train takes a cost tensor to minimize. Trains one batch. Returns the
    // average cost as a Scalar.
    if(printInfo){
        for(let i = 0; i <= 100; i++){
            let input = model.x.shape.length === 1 ?
                Array1D.new(ds[i].input):
                Array2D.new(model.x.shape, ds[i].input);
            let target = ds[i].output; 
            let pred = session.eval(
                model.output,
                [{tensor: model.x, data: input}]
            );
            try {
                console.log("p:", pred.dataSync(), "t:", target);
            } catch (e){
                console.log("Error at dataSync", e);
            }
        }
    }

    let [runtime, feed] = model.session.trainModStart(
        model.cost, feedEntries, batchSize, optimizer)

    let data = model.math.backend.data;
    if(modelByBatch && optimizerParams !== null){
        setOptimizerParams(optimizer, optimizerParams, data);
    }
    console.log(getOptimizerParams(optimizer, data))

    const cost = model.session.trainModEnd(
        runtime, feed, model.cost, batchSize, optimizer, CostReduction.MEAN)

    try { costVal = await cost.val(); }
    catch(e){console.log("Error at await", e);}

    console.log('last average cost (' + iter + '): ' + costVal);

    /*
    if (modelByBatch){
        modelParams.session = model.session;
        modelParams.graph= model.graph ;
        model.graph.nodes.forEach((node) => {
            if (
                (node.data !== undefined) && (! node.data.isDisposed)
            ){ node.data.dispose();}
        })
        model.graph.nodes = model.graph.nodes.slice(0, 0);
    }
    */

    return([feedEntries, optimizer, model]);
}


// END of export boiler plate
exports.doBatch = doBatch;
exports.getOptimizerParams = getOptimizerParams;
exports.setOptimizerParams = setOptimizerParams;
})(
    typeof exports === 'undefined'?  this['deep_batch_train']={}: exports
);



