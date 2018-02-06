// Start of module boilerplate, insures that code is useable both
// on server side and client side
(function(exports){

function getOptimizer(optimizerType, learningRate, momentum){

    let optimizer = null;

    switch (optimizerType){
        case 'momentum':
            optimizer = new MomentumOptimizer(learningRate, momentum);
            break;
        case 'SGD':
            optimizer = new SGDOptimizer(learningRate);
            break;
        case 'RMSProp':
            optimizer = new RMSPropOptimizer(learningRate, momentum);
            break;
        case 'Adam':
            optimizer = new AdamOptimizer(learningRate, momentum, 0.999);
            break;
        case 'Adagrad':
            optimizer = new AdagradOptimizer(learningRate, momentum);
            break;
    }

    return(optimizer)
}

function prepareFeed(model, xProvider, lProvider){

    // return model.math.scope(() => {
        // Maps tensors to InputProviders.
        xFeed = {tensor: model.x, data: xProvider};
        lFeed = {tensor: model.y, data: lProvider};

        feedEntries = [xFeed, lFeed];

        return(feedEntries);
    // });
}


async function doBatch(
    batchSize,
    debug,
    ds,
    feedEntries,
    feeder,
    graph,
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
    xProvider
){

    // we can chose to keep the same model / optimizer accross batches
    // or
    // we can chose to have a new model / optimizer for each batch
    if(modelByBatch){
        model = getDeepModel(modelParams);
        feeder.math = model.math;
        [xProvider, lProvider] = feeder.next();
    }

    // when setting optimizer_by_batch = false with
    // model_by_btach = false ==> error, optimizer by batch probably
    // keeps old refs
    if (optimizerByBatch){
        optimizer = getOptimizer(optimizerType, learningRate, momentum);
    }

    debug.xProvider = xProvider;
    debug.lProvider = lProvider;
    debug.model = model;
    debug.optimizer = optimizer;

    if (feedEntries === null){
        feedEntries = prepareFeed(model, xProvider, lProvider);
    }

    // Train takes a cost tensor to minimize. Trains one batch. Returns the
    // average cost as a Scalar.
    if(printInfo){
        for(let i = 0; i <= 100; i++){
            let input = model.x.shape.length === 1 ?
                Array1D.new(ds[i].input):
                Array2D.new(model.x.shape, ds[i].input);
            let target = ds[i].output; 
            let pred = session.eval(
                model.output, [{tensor: model.x, data: input}]
            );
            try {
                console.log("p:", pred.dataSync(), "t:", target);
            } catch (e){
                console.log("Error at dataSync", e);
            }
        }
    }

    const cost = model.session.trainMod(
        model.cost, feedEntries, batchSize, optimizer,
        CostReduction.MEAN);

    try {
        costVal = await cost.val();
    } catch(e){
        console.log("Error at await", e);
    }
    console.log('last average cost (' + i + '): ' + costVal);

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

    return([feedEntries, optimizer])
}


// END of export boiler plate
exports.doBatch = doBatch;
})(
    typeof exports === 'undefined'?  this['worker']={}: exports
);



onconnect = function(e) {



}



