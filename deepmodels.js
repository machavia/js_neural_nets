if(typeof(require) === 'function'){

    const {
        ENV,
        AdagradOptimizer,
        AdamOptimizer,
        AdamaxOptimizer,
        Array1D,
        Array2D,
        Array3D,
        CostReduction,
        FeedEntry,
        GPGPUContext,
        Graph,
        MomentumOptimizer,
        RMSPropOptimizer,
        Scalar,
        Session,
        SGDOptimizer,
        Tensor,
        util,
        NDArray,
        NDArrayMath,
        InCPUMemoryShuffledInputProviderBuilder,
        InGPUMemoryShuffledInputProviderBuilder
    } = require('./deeplearn')

    // require('./node_modules/deeplearn/dist/deeplearn')

    const {
        Initializer, VarianceScalingInitializer, ZerosInitializer,
        NDArrayInitializer
    } = require('./deeplearn')
    // require('./node_modules/deeplearn/dist/deeplearn');

}else{
    
    ENV = deeplearn.ENV;
    AdagradOptimizer = deeplearn.AdagradOptimizer;
    AdamOptimizer = deeplearn.AdamOptimizer;
    AdamaxOptimizer = deeplearn.AdamaxOptimizer;
    Array1D = deeplearn.Array1D;
    Array2D = deeplearn.Array2D;
    Array3D = deeplearn.Array3D;
    CostReduction = deeplearn.CostReduction;
    FeedEntry = deeplearn.FeedEntry;
    GPGPUContext = deeplearn.GPGPUContext;
    Graph = deeplearn.Graph;
    MomentumOptimizer = deeplearn.MomentumOptimizer;
    RMSPropOptimizer = deeplearn.RMSPropOptimizer;
    Scalar = deeplearn.Scalar;
    Session = deeplearn.Session;
    SGDOptimizer = deeplearn.SGDOptimizer;
    Tensor = deeplearn.Tensor;
    util = deeplearn.util;
    NDArray = deeplearn.NDArray;
    NDArrayMath = deeplearn.NDArrayMath;
    InCPUMemoryShuffledInputProviderBuilder = deeplearn.InCPUMemoryShuffledInputProviderBuilder;
    InCPUMemoryShuffledInputProviderBuilderExtended = deeplearn.InCPUMemoryShuffledInputProviderBuilderExtended;
    InGPUMemoryShuffledInputProviderBuilder = deeplearn.InGPUMemoryShuffledInputProviderBuilder;
    Initializer = deeplearn.Initializer;
    VarianceScalingInitializer = deeplearn.VarianceScalingInitializer;
    ZerosInitializer = deeplearn.ZerosInitializer;
    NDArrayInitializer = deeplearn.NDArrayInitializer;

}


// Start of module boilerplate, insures that code is useable both
// on server side and client side
(function(exports){

/* stolen from graph/graph.ts + modified */
function addDense(
    math,
    graph, name, x, units,
    activation, useBias = true,
    kernelInitValue = null,
    biasInitValue = null,
    kernelVariable = null,
    biasVariable = null
){
    let [kernelInitializer, biasInitializer] = [null, null];
    if (kernelInitValue === null){
         kernelInitializer = math.scope(() => {
             return(new VarianceScalingInitializer())});
    }else{
         kernelInitializer = math.scope(() => {
             return(new NDArrayInitializer(
                Array2D.new([x.shape[0], units], kernelInitValue)))});
    }
    if (biasInitValue === null){
        biasInitializer = math.scope(() => {return(new ZerosInitializer())});
    }else{
        biasInitializer = math.scope(() => {
            return(new NDArrayInitializer(
                Array1D.new(biasInitValue)))});
    }
    const weights = kernelVariable === null ?
        graph.variable(
            name + '-weights',
            math.scope(() => {
                return(kernelInitializer.initialize(
                    [x.shape[0], units], x.shape[0], units
                ));
            })
        ) :
        kernelVariable;

    let lin;
    let out;
    let toReturn = [];
    const mm = math.scope(() => {return(graph.matmul(x, weights))});

    toReturn.push(weights);

    if (useBias) {
        const bias = biasVariable === null ?
            graph.variable(
                name + '-bias',
                math.scope(() => {
                    return(
                        biasInitializer.initialize([units], x.shape[0], units)
                    );
                })
            ) : biasVariable;
        lin = graph.add(mm, bias);
        toReturn.push(bias);
    }else{
        lin = mm;
    }

    if (activation != null) {
        out = activation(lin);
    }

    return([toReturn, out]);
}

class FFN1D {
    constructor({
        graph, input_size=2, hidden_size=3, nPredictions=1
    }){
        this.input_size = input_size;
        this.hidden_size = hidden_size;

        this.x = graph.placeholder('x', [input_size]);
        this.y = graph.placeholder('x', [nPredictions]);

        const EPSILON = 1e-7;
        // Variables are containers that hold a value that can be updated from
        // trainingraph.
        // Here we initialize the multiplier variable randomly.
        const hiddenLayer = graph.layers.dense(
            'hiddenLayer', this.x, hidden_size, (x) => graph.relu(x), true
        );
        this.output = graph.layers.dense(
            'outputLayer', hiddenLayer, nPredictions, (x) => graph.sigmoid(x),
            true
        );
        this.cost = graph.reduceSum(graph.add(
            graph.multiply(
                graph.constant([-1]),
                graph.multiply(
                    this.y,
                    graph.log(
                        graph.add(this.output, graph.constant([EPSILON]))))),
            graph.multiply(
                graph.constant([-1]),
                graph.multiply(
                    graph.subtract(graph.constant([1]), this.y),
                    graph.log(
                        graph.add(
                            graph.subtract(graph.constant([1]), this.output),
                            graph.constant([EPSILON])))))));

        console.log("nPredictions:", nPredictions);

        // WARNING: placeholder are not included in the graph apparently so
        // evaluating them will throw an error !!!

    }

}

class LSTMCell{
    constructor({
        graph, input_size=2, hidden_size=3, nPredictions=1,
        x=null, h_tm1=null, c_tm1=null, add_cost=true, weights=null
    }){
        this.weights = weights;
        this.input_size = input_size;
        this.hidden_size = hidden_size;

        let bi_i, bf_i, bc_i, bo_i, bout_i;
        let Wi_i, Wf_i, Wc_i, Wo_i, Wout_i;

        let Wi, bi, Wf, bf, Wc, bc, Wo, bo, Wout, bout;

        if (weights !== null){
            //
            [[Wi, bi], [Wf, bf], [Wc, bc], [Wo, bo], [Wout, bout]] = [
                weights['i'], weights['f'], weights['c'], weights['o'],
                weights['out']
            ];

            let setValues = [];
            let vars = [Wi, bi, Wf, bf, Wc, bc, Wo, bo, Wout, bout];
            for (let ival of vars){
                let ndarrayData = new NDArrayInitializer(ival);
                setValues.push(ndarrayData);
            }

            [
                Wi_i, bi_i, 
                Wf_i, bf_i, 
                Wc_i, bc_i,
                Wo_i, bo_i,
                Wout_i, bout_i
            ] = setValues;

        }else{
            let b_init = new ZerosInitializer();
            [bi_i, bf_i, bc_i, bo_i, bout_i] = [
                b_init, b_init, b_init, b_init, b_init 
            ];

            let W_init = new VarianceScalingInitializer();
            [Wi_i, Wf_i, Wc_i, Wo_i, Wout_i] = [
                W_init, W_init, W_init, W_init, W_init 
            ];
        }

        this.x = graph.placeholder('x', [input_size]);
        if (h_tm1 === null){
            this.h_tm1 = graph.placeholder('h_tm1', [hidden_size]);
            this.c_tm1 = graph.placeholder('c_tm1', [hidden_size]);
        }else{
            this.h_tm1 = h_tm1;
            this.c_tm1 = c_tm1;
        }

        let XHC_prev = graph.concat1d(
            this.x, graph.concat1d(this.h_tm1, this.c_tm1)
        );

        let XH = graph.concat1d(this.x, this.h_tm1);

        [[this.Wi, this.bi], this.i] = addDense(
            math,
            graph,
            'i', XHC_prev, hidden_size,
            (x) => graph.sigmoid(x), true,
            Wi_i, bi_i
            );
             // graph.layers.dense

        [[this.Wf, this.bf], this.f] = addDense(
            math,
            graph,
            'f', XHC_prev, hidden_size,
            (x) => graph.sigmoid(x), true,
            Wf_i, bf_i
            );
            // graph.layers.dense

        let half_c;
        [[this.Wc, this.bc], half_c] = addDense(
            math,
            graph,
            'c_half', XH, hidden_size,
            (x) => graph.tanh(x), true,
            Wc_i, bc_i
            )
            //graph.layers.dense
        this.c = graph.add(
            graph.multiply(this.f, this.c_tm1),
            graph.multiply(this.i, half_c)
        );

        let XHC = graph.concat1d(
            this.x, graph.concat1d(this.h_tm1, this.c)
        );

        [[this.Wo, this.bo], this.o] = addDense(
            math,
            graph,
            'o', XHC, hidden_size,
            (x) => graph.sigmoid(x), true,
            Wo_i, bo_i
            );
            // graph.layers.dense

        // define h
        this.h = graph.multiply(this.o, graph.tanh(this.c));

        [[this.Wout, this.bout], this.output] = addDense(
            math,
            graph,
            'classif', this.h, nPredictions,
            (x) => graph.sigmoid(x), true,
            Wout_i, bout_i
            );
            // graph.layers.dense

        if (add_cost){
            this.y = graph.placeholder('y', [nPredictions]);
            const EPSILON = 1e-7;
            this.cost = graph.reduceSum(graph.add(
                graph.multiply(
                    graph.constant([-1]),
                    graph.multiply(
                        this.y,
                        graph.log(
                            graph.add(
                                this.output, graph.constant([EPSILON]))))),
                graph.multiply(
                    graph.constant([-1]),
                    graph.multiply(
                        graph.subtract(graph.constant([1]), this.y),
                        graph.log(
                            graph.add(
                                graph.subtract(
                                    graph.constant([1]), this.output),
                                graph.constant([EPSILON])))))));
        }

        console.log("nPredictions:", nPredictions)

        this.weights = {
            'i': [this.Wi, this.bi],
            'f': [this.Wf, this.bf],
            'c': [this.Wc, this.bc],
            'o': [this.Wo, this.bo],
            'out': [this.Wout, this.bout]
        };

    }

    getWeightsValuesSync(session){
        let weights = {};
        for (let [k, v] of Object.entries(this.weights)){
            let w_b = session.evalAll(v);
            let val = []
            for (let x of w_b){
                let ar;
                if (x.shape.length > 1){
                    ar = new Array2D.new(x.shape, x.dataSync());
                }else{
                    ar = new Array1D.new(x.dataSync());
                }
                val.push(ar);
            }
            weights[k] = val;
        }
        return(weights);
    }

}


class LSTMCellShared{
    constructor({
        math,
        graph, input_size=2, hidden_size=3, nPredictions=1,
        x=null, h_tm1=null, c_tm1=null, add_cost=true,
        sharedVariables=null,
        weightValues=null
    }){
        this.math = math;
        this.graph = graph;
        this.weights = sharedVariables;
        this.input_size = input_size;
        this.hidden_size = hidden_size;
        this.hasCost = add_cost;

        let bi_i, bf_i, bc_i, bo_i, bout_i;
        let Wi_i, Wf_i, Wc_i, Wo_i, Wout_i;

        let Wi, bi, Wf, bf, Wc, bc, Wo, bo, Wout, bout;

        if (this.weights !== null){
            // set variables (i.e. containers)
            [[Wi, bi], [Wf, bf], [Wc, bc], [Wo, bo], [Wout, bout]] = [
                this.weights['i'],
                this.weights['f'],
                this.weights['c'],
                this.weights['o'],
                [null, null] // shared variables except for output
            ];

            // set initial values
            let setValues = [];
            let vars = [Wi, bi, Wf, bf, Wc, bc, Wo, bo, Wout, bout];

        }else{
            [
                Wi, bi, 
                Wf, bf, 
                Wc, bc,
                Wo, bo,
                Wout, bout
            ] = [
                null, null,
                null, null,
                null, null,
                null, null,
                null, null
            ];
        }
        
        if(weightValues !== null){
            [
                Wi_i, bi_i, 
                Wf_i, bf_i, 
                Wc_i, bc_i,
                Wo_i, bo_i,
                Wout_i, bout_i
            ] = weightValues;
        }
        else{
            [
                Wi_i, bi_i,
                Wf_i, bf_i,
                Wc_i, bc_i,
                Wo_i, bo_i,
                Wout_i, bout_i
            ] = [
                null, null,
                null, null,
                null, null,
                null, null,
                null, null   
            ];

        }

        this.x = x === null ?
            graph.placeholder('x', [input_size]):
            x;

        this.x_check = graph.reshape(x, x.shape);

        if (h_tm1 === null){
            this.h_tm1 = graph.placeholder('h_tm1', [hidden_size]);
            this.c_tm1 = graph.placeholder('c_tm1', [hidden_size]);
        }else{
            this.h_tm1 = h_tm1;
            this.c_tm1 = c_tm1;
        }

        let XHC_prev = graph.concat1d(
            this.x, graph.concat1d(this.h_tm1, this.c_tm1)
        );

        let XH = graph.concat1d(this.x, this.h_tm1);

        [[this.Wi, this.bi], this.i] = addDense(
            math,
            graph,
            'i', XHC_prev, hidden_size,
            (x) => graph.sigmoid(x), true,
            Wi_i, bi_i, Wi, bi
            );
             // graph.layers.dense

        [[this.Wf, this.bf], this.f] = addDense(
            math,
            graph,
            'f', XHC_prev, hidden_size,
            (x) => graph.sigmoid(x), true,
            Wf_i, bf_i, Wf, bf
            );
            // graph.layers.dense

        let half_c;
        [[this.Wc, this.bc], half_c] = addDense(
            math,
            graph,
            'c_half', XH, hidden_size,
            (x) => graph.tanh(x), true,
            Wc_i, bc_i, Wc, bc
            )
            //graph.layers.dense
        this.c = graph.add(
            graph.multiply(this.f, this.c_tm1),
            graph.multiply(this.i, half_c)
        );

        let XHC = graph.concat1d(
            this.x, graph.concat1d(this.h_tm1, this.c)
        );

        [[this.Wo, this.bo], this.o] = addDense(
            math,
            graph,
            'o', XHC, hidden_size,
            (x) => graph.sigmoid(x), true,
            Wo_i, bo_i, Wo, bo 
            );
            // graph.layers.dense

        // define h
        this.h = graph.multiply(this.o, graph.tanh(this.c));

        // [this.Wout, this.bout] = [null, null];

        if (add_cost){

            [[this.Wout, this.bout], this.output] = addDense(
                math,
                graph,
                'classif', this.h, nPredictions,
                (x) => graph.sigmoid(x), true,
                Wout_i,bout_i, Wout, bout
                );
                // graph.layers.dense
            //
            const EPSILON = 1e-7;

            const arrayOne = math.scope(() => {return(Array1D.new([1]));})
            const arrayMOne = math.scope(() => {return(Array1D.new([-1]));})
            const eps = math.scope(() => {return(Array1D.new([EPSILON]));})

            this.y = graph.placeholder('y', [nPredictions]);
            this.cost = graph.reduceSum(graph.add(
                    graph.multiply(
                        graph.constant(arrayMOne),
                        graph.multiply(
                            this.y,
                            graph.log(
                                graph.add(
                                    this.output,
                                    graph.constant(eps))))),
                    graph.multiply(
                        graph.constant(arrayMOne),
                        graph.multiply(
                            graph.subtract(
                                graph.constant(arrayOne), this.y),
                            graph.log(
                                graph.add(
                                    graph.subtract(
                                        graph.constant(arrayOne), this.output),
                                    graph.constant(eps)))))));
        }

        console.log("nPredictions:", nPredictions)

        this.weights = {
            'i': [this.Wi, this.bi],
            'f': [this.Wf, this.bf],
            'c': [this.Wc, this.bc],
            'o': [this.Wo, this.bo],
            'out': [this.Wout, this.bout]
        };

    }

    getWeightsValuesSync(session){
        let weights = {};
        for (let [k, v] of Object.entries(this.weights)){
            let w_b = session.evalAll(v);
            let val = []
            for (let x of w_b){
                let ar;
                if (x.shape.length > 1){
                    ar = new Array2D.new(x.shape, x.dataSync());
                }else{
                    ar = new Array1D.new(x.dataSync());
                }
                val.push(ar);
            }
            weights[k] = val;
        }
        return(weights);
    }

    getWeightsValues(){
        let data = this.math.backend.data;
        let weights = [];
        let layers = ['i', 'f', 'c', 'o'];
        if (this.hasCost){
            layers.push('out');
        }
        for(let layer of layers){
            // push weight mat
            weights.push(
                data[this.weights[layer][0].node.data.dataId]);
            // push bias
            weights.push(
                data[this.weights[layer][1].node.data.dataId]);
        }
        if (! this.hasCost){
            weights.push(null);
            weights.push(null);
        }
        return(weights);
    }

}

class RNNShared{

    constructor({
        math, graph, session, inputSize=2, hiddenSize=3, nPredictions=1,
        cellClass=LSTMCellShared, seqLength=10,
        sharedVariables=null,
        weightValues=null
    }){

        // this.h_t0 = graph.placeholder('h_t0', [hiddenSize]);
        // this.c_t0 = graph.placeholder('c_t0', [hiddenSize]);
        this.h_t0 = graph.constant(Array1D.zeros([hiddenSize]));
        this.c_t0 = graph.constant(Array1D.zeros([hiddenSize]));
        this.x = graph.placeholder('x', [seqLength, inputSize]);
        this.xs = [];
        this.cells = [];
        this.sharedVariables = sharedVariables;
        this.sharedWeightValues = weightValues;
        let [c_tm1, h_tm1] = [null, null];

        for(let i = 0; i < seqLength; i++){
            /*
            TODO ok a bit dirty: in absence of a gather or slice
            op we multiply x by a indicator vector with size=n_rows(x)
            with one at the index we wish to keep for step, zeros elsewhere...
            */
            let indicator = Array1D.zeros([this.x.shape[0]]);
            indicator.set(1, i);
            let indicatorTensor = graph.constant(indicator);
            this.xs.push(graph.matmul(indicatorTensor, this.x));
            // shared parameters ?
            // this.cell
            let add_cost = false;
            if (i === seqLength - 1){
                add_cost = true;
            }
            if (i === 0){
                c_tm1 = this.c_t0;
                h_tm1 = this.h_t0;
            }

            let cell = new cellClass({
                math,
                graph, input_size: inputSize, hidden_size: hiddenSize,
                nPredictions: nPredictions,
                h_tm1: h_tm1, c_tm1: c_tm1,
                add_cost: add_cost,
                sharedVariables: this.sharedVariables,
                weightValues: null,
                x: this.xs[i]
            });

            c_tm1 = cell.c;
            h_tm1 = cell.h;

            if (add_cost){
                // cost, target will be the same as last cell
                this.y = cell.y,
                this.output = cell.output;
                this.cost = cell.cost;
            }

            // will keep weight of the last cell to date
            //if (this.sharedVariables === null){
            this.sharedVariables = cell.weights;
            this.sharedWeightValues = cell.getWeightsValues();
            // }

            // console.log(this.sharedVariables)
            // console.log(this.sharedWeightValues)

            this.cells.push(cell)
        }
    }

    getWeightsValues(){
        let weights = this.cells[this.cells.length - 1].getWeightsValues();
        return( weights );
    }
}

class Feeder{

    constructor({
        ds, dim, seqLen, outputSize, hiddenSize, make2d, batchSize,
        addState=false, math=null
    }){
        this.ds = ds;
        this.dim = dim;
        this.seqLen = seqLen;
        this.outputSize = outputSize;
        this.hiddenSize = hiddenSize;
        this.make2d = make2d;
        this.batchSize = batchSize;
        this.index = 0;
        this.addState = addState;
        this.math = math;
        this.allocated = [];
    }

    next(){

        // return(this.math.scope(() => {

            let [x, h, c, labels] = [[], [], [], []];

            let labelTensor = [];
            let costTensor = [];
            let optimizer = [];
            let feedEntries = [];

            let shuffledInputProviderBuilder = [];
            let [xProvider, hProvider, cProvider, lProvider] = []
            let [xFeed, hFeed, cFeed, lFeed] = []

            for(let curr = 0; curr < this.batchSize; curr++){
                if (this.index == this.ds.length){this.index = 0;}
                let dat = this.ds[this.index];

                let [xItem, lItem, hItem, cItem] = [null, null, null, null]

                if (this.make2d){

                    xItem = Array2D.new([this.seqLen, this.dim], dat.input);
                    lItem = Array1D.new(dat.output);

                }else{

                    xItem = Array1D.new(dat.input);
                    lItem = Array1D.new(dat.output);

                    if(this.addState){
                        hItem = Array1D.zeros([this.hiddenSize]);
                        cItem = Array1D.zeros([this.hiddenSize]);
                    }
                }

                this.allocated.push(xItem);
                this.allocated.push(lItem);
                x.push(xItem);
                labels.push(lItem);

                if (this.addState){
                    this.allocated.push(hItem);
                    this.allocated.push(cItem);
                    h.push(hItem);
                    c.push(cItem);
                }

                this.index++;
            }

            let toGet = this.addState ? [x, h, c, labels] : [x, labels];

            // Shuffles inputs and labels and keeps them mutually in sync.
            shuffledInputProviderBuilder =
                new InCPUMemoryShuffledInputProviderBuilderExtended(toGet);

            // debug.shuffledInputProvider = shuffledInputProviderBuilder;

            let toReturn = 
                shuffledInputProviderBuilder.getInputProviders();

            // return(toReturn);

            return(toReturn);
        // }))
    }

}

function getDeepModel({
    modelType='RNNLSTM',
    nPredictions=outputSize,
    hiddenSize=hiddenSize,
    inputSize=1,
    seqLength=seqLength,
    init_weights=null,
    math=null,
    graph=null,
    session=null
}){

    // TODO set safe to true here
    if (math === null){
        deeplearn.ENV.setMath(new deeplearn.NDArrayMath('cpu', false))
        math = ENV.math;
    }
    if (graph === null){
        graph = new Graph();
    }
    if (session === null){
        session = new Session(graph, math);
    }

    let model = null;

    switch (modelType){
        case 'RNNLSTM':
            model = new RNNShared({
                math: math, graph: graph, session: session, nPredictions: nPredictions,
                hiddenSize: hiddenSize, inputSize: inputSize,
                seqLength: seqLength,
                weightValues: null
            });
            break;
        case 'LSTM':
            // TODO check that seqLength == 1
            model = new RNNShared({
                graph: graph, session: session, nPredictions: outputSize,
                hiddenSize: hiddenSize, inputSize: inputSize,
                seqLength: 1,
                weightValues: null
            });
            break;
        case 'FFN1D':
            model = new FFN1D({
                graph: graph, input_size: inputSize, hidden_size: hiddenSize,
                nPredictions: nPredictions
            })
    }

    model.session = session;
    model.graph = graph;
    model.math = math;

    return(model);
}


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

function dsToDeepDS({
    ds, dim=1, flatten=true, seqLen=1, outputSize=2,
    hiddenSize=8, make2d=false
}){
    
    let [x, h, c, labels] = [[], [], [], []];

    let labelTensor = [];
    let costTensor = [];
    let optimizer = [];
    let feedEntries = [];

    let shuffledInputProviderBuilder = [];
    let [xProvider, hProvider, cProvider, lProvider] = []
    let [xFeed, hFeed, cFeed, lFeed] = []

    for(dat of ds){
        if (make2d){
            x.push(Array2D.new([seqLen, dim], dat.input));
            labels.push(Array1D.new(dat.output));
            h.push(Array2D.zeros([seqLen, hiddenSize]));
            c.push(Array2D.zeros([seqLen, hiddenSize]));
        }else{
            x.push(Array1D.new(dat.input));
            labels.push(Array1D.new(dat.output));
            h.push(Array1D.zeros([hiddenSize]));
            c.push(Array1D.zeros([hiddenSize]));
        }
    }

    // Shuffles inputs and labels and keeps them mutually in sync.
    shuffledInputProviderBuilder =
        new InCPUMemoryShuffledInputProviderBuilderExtended([x, h, c, labels]);

    // debug.shuffledInputProvider = shuffledInputProviderBuilder;

    [xProvider, hProvider, cProvider, lProvider] =
        shuffledInputProviderBuilder.getInputProviders();

    return([ds, xProvider, hProvider, cProvider, lProvider]);
}

// END of export boiler plate
exports.FFN1D = FFN1D;
exports.LSTMCell = LSTMCell;
exports.LSTMCellShared = LSTMCellShared;
exports.RNNShared = RNNShared;
exports.getDeepModel = getDeepModel;
exports.Feeder = Feeder; // TODO should be in a "data" module
exports.getOptimizer= getOptimizer;
exports.prepareFeed = prepareFeed;
exports.dsToDeepDS = dsToDeepDS;

})(
    typeof exports === 'undefined'?  this['deepmodels']={}: exports
);


