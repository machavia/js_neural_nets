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

async function doBatchFromScratch(
    batchSize,
    feeder,
    iter,
    learningRate,
    model,
    modelParams,
    momentum,
    optimizerType,
    optimizerParams=null
){

    let model = getDeepModel(modelParams);
    feeder.math = model.math;
    [xProvider, lProvider] = feeder.next();

    optimizer = getOptimizer(optimizerType, learningRate, momentum);

    let feedEntries = prepareFeed(model, xProvider, lProvider);

    let [runtime, feed] = model.session.trainModStart(
        model.cost, feedEntries, batchSize, optimizer)

    let data = model.math.backend.data;

    if(optimizerParams !== null){
        setOptimizerParams(optimizer, optimizerParams, data);}

    const cost = model.session.trainModEnd(
        runtime, feed, model.cost, batchSize, optimizer, CostReduction.MEAN)

    try { costVal = await cost.val(); }
    catch(e){console.log("Error at await", e);}

    console.log('last average cost (' + iter + '): ' + costVal);

    return([feedEntries, optimizer, model]);
}

async function doBatch({
    batchSize,
    feedEntries,
    iter,
    model,
    momentum,
    optimizer,
    session,
}){


    let [runtime, feed] = model.session.trainModStart(
        model.cost, feedEntries, batchSize, optimizer)

    let data = model.math.backend.data;

    const cost = model.session.trainModEnd(
        runtime, feed, model.cost, batchSize, optimizer, CostReduction.MEAN)

    try { costVal = await cost.val(); }
    catch(e){console.log("Error at await", e);}

    console.log('last average cost (' + iter + '): ' + costVal);

    /*
    if (modelByBatch){
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

async function doCost(model, ds, session){
    // Train takes a cost tensor to minimize. Trains one batch. Returns the
    // average cost as a Scalar.
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



// END of export boiler plate
exports.doBatch = doBatch;
exports.getOptimizerParams = getOptimizerParams;
exports.setOptimizerParams = setOptimizerParams;
})(
    typeof exports === 'undefined'?  this['deep_batch_train']={}: exports
);



