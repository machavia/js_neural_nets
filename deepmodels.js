// Start of module boilerplate, insures that code is useable both
// on server side and client side
(function(exports){

/* stolen from graph/graph.ts + modified */
function addDense(
    graph, name, x, units,
    activation, useBias = true,
    kernelInitializer = null,
    biasInitializer = null,
    kernelVariable = null,
    biasVariable = null
){
    if (kernelInitializer === null){
         kernelInitializer = new VarianceScalingInitializer();
    }
    if (biasInitializer === null){
        biasInitializer = new ZerosInitializer();
    }
    const weights = kernelVariable === null ?
        graph.variable(
            name + '-weights',
            kernelInitializer.initialize(
                [x.shape[0], units], x.shape[0], units
            )
        ) :
        kernelVariable;

    let lin;
    let out;
    let toReturn = [];
    const mm = graph.matmul(x, weights);

    toReturn.push(weights);

    if (useBias) {
        const bias = biasVariable === null ?
            graph.variable(
                name + '-bias',
                biasInitializer.initialize([units], x.shape[0], units)):
            biasVariable;
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
            graph,
            'i', XHC_prev, hidden_size,
            (x) => graph.sigmoid(x), true,
            Wi_i, bi_i
            );
             // graph.layers.dense

        [[this.Wf, this.bf], this.f] = addDense(
            graph,
            'f', XHC_prev, hidden_size,
            (x) => graph.sigmoid(x), true,
            Wf_i, bf_i
            );
            // graph.layers.dense

        let half_c;
        [[this.Wc, this.bc], half_c] = addDense(
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
            graph,
            'o', XHC, hidden_size,
            (x) => graph.sigmoid(x), true,
            Wo_i, bo_i
            );
            // graph.layers.dense

        // define h
        this.h = graph.multiply(this.o, graph.tanh(this.c));

        [[this.Wout, this.bout], this.output] = addDense(
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

    getWeights(session){
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
        graph, input_size=2, hidden_size=3, nPredictions=1,
        x=null, h_tm1=null, c_tm1=null, add_cost=true, sharedVariables=null
    }){
        this.sharedVariables = sharedVariables;
        this.input_size = input_size;
        this.hidden_size = hidden_size;

        let bi_i, bf_i, bc_i, bo_i, bout_i;
        let Wi_i, Wf_i, Wc_i, Wo_i, Wout_i;

        let Wi, bi, Wf, bf, Wc, bc, Wo, bo, Wout, bout;

        if (this.sharedVariables !== null){
            //
            [[Wi, bi], [Wf, bf], [Wc, bc], [Wo, bo], [Wout, bout]] = [
                this.sharedVariables['i'], this.sharedVariables['f'],
                this.sharedVariables['c'], this.sharedVariables['o'],
                this.sharedVariables['out']
            ];

            let setValues = [];
            let vars = [Wi, bi, Wf, bf, Wc, bc, Wo, bo, Wout, bout];

            [
                Wi_i, bi_i, 
                Wf_i, bf_i, 
                Wc_i, bc_i,
                Wo_i, bo_i,
                Wout_i, bout_i
            ] = setValues;

        }else{
            [bi_i, bf_i, bc_i, bo_i, bout_i] = [
                null, null, null, null, null 
            ];

            [Wi_i, Wf_i, Wc_i, Wo_i, Wout_i] = [
                null, null, null, null, null 
            ];
        }

        this.x = x === null ?
            graph.placeholder('x', [input_size]):
            x;

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
            graph,
            'i', XHC_prev, hidden_size,
            (x) => graph.sigmoid(x), true,
            null, null, Wi_i, bi_i
            );
             // graph.layers.dense

        [[this.Wf, this.bf], this.f] = addDense(
            graph,
            'f', XHC_prev, hidden_size,
            (x) => graph.sigmoid(x), true,
            null, null, Wf_i, bf_i
            );
            // graph.layers.dense

        let half_c;
        [[this.Wc, this.bc], half_c] = addDense(
            graph,
            'c_half', XH, hidden_size,
            (x) => graph.tanh(x), true,
            null, null, Wc_i, bc_i
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
            graph,
            'o', XHC, hidden_size,
            (x) => graph.sigmoid(x), true,
            null, null, Wo_i, bo_i
            );
            // graph.layers.dense

        // define h
        this.h = graph.multiply(this.o, graph.tanh(this.c));

        [this.Wout, this.bout] = [null, null];

        if (add_cost){

            [[this.Wout, this.bout], this.output] = addDense(
                graph,
                'classif', this.h, nPredictions,
                (x) => graph.sigmoid(x), true,
                null, null, null, null 
                );
                // graph.layers.dense

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

}

class RNNShared{

    constructor({
        graph, session, inputSize=2, hiddenSize=3, nPredictions=1,
        cellClass=LSTMCellShared, seqLength=10, init_weights=null,
        sharedVariables=null
    }){

        // this.h_t0 = graph.placeholder('h_t0', [hiddenSize]);
        // this.c_t0 = graph.placeholder('c_t0', [hiddenSize]);
        this.h_t0 = graph.constant(Array1D.zeros([hiddenSize]));
        this.c_t0 = graph.constant(Array1D.zeros([hiddenSize]));
        this.x = graph.placeholder('x', [seqLength, inputSize]);
        this.xs = [];
        this.cells = [];
        this.sharedVariables = init_weights;
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
                graph, input_size: inputSize, hidden_size: hiddenSize,
                nPredictions: nPredictions,
                h_tm1: h_tm1, c_tm1: c_tm1,
                add_cost: add_cost,
                sharedVariables: this.sharedVariables,
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

            if (this.sharedVariables === null){
                this.sharedVariables = cell.sharedVariables;
            }

            this.cells.push(cell)
        }
    }
}

function dsToDeepDS({
    ds, dim=1, flatten=true, seq_len=1, outputSize=2,
    hiddenSize=8, noise=0, make2d=false
}){
    
    let [x, h, c, labels] = [[], [], [], []];

    math = ENV.math;
    const graph = new Graph();
    const session = new Session(graph, math);

    let labelTensor = [];
    let costTensor = [];
    let optimizer = [];
    let feedEntries = [];

    let shuffledInputProviderBuilder = [];
    let [xProvider, hProvider, cProvider, lProvider] = []
    let [xFeed, hFeed, cFeed, lFeed] = []

    for(dat of ds){
        if (make2d){
            x.push(Array2D.new([2 * seq_len, dim], dat.input));
            labels.push(Array1D.new(dat.output));
            h.push(Array2D.zeros([2 * seq_len, hiddenSize]));
            c.push(Array2D.zeros([2 * seq_len, hiddenSize]));
        }else{
            x.push(Array1D.new(dat.input));
            labels.push(Array1D.new(dat.output));
            h.push(Array1D.zeros([hiddenSize]));
            c.push(Array1D.zeros([hiddenSize]));
        }
    }

    // Shuffles inputs and labels and keeps them mutually in sync.
    shuffledInputProviderBuilder =
        new InCPUMemoryShuffledInputProviderBuilder([x, h, c, labels]);

    [xProvider, hProvider, cProvider, lProvider] =
        shuffledInputProviderBuilder.getInputProviders();

    return([ds, xProvider, hProvider, cProvider, lProvider]);
}

async function deepTrain({
    model=null,
    printInfo=false,
    dsProviders=null,
    batchSize = 64,
    learningRate = 0.1,
    momentum = 0.9,
}){

    // await firstLearn();
    graph = model.graph;
    session = model.session;

    // Maps tensors to InputProviders.
    xFeed = {tensor: model.x, data: xProvider};
    lFeed = {tensor: model.y, data: lProvider};

    feedEntries = [xFeed, lFeed];

    x_check = graph.reshape(model.x, model.x.shape);
    l_check = graph.reshape(model.y, model.y.shape);

    optimizer = new MomentumOptimizer(learningRate, momentum);
    // optimizer = new RMSPropOptimizer(learningRate, momentum);
    // optimizer = new SGDOptimizer(learningRate);

    NUM_BATCHES = 1000;
    for (let i = 0; i < NUM_BATCHES; i++) {
        // Train takes a cost tensor to minimize. Trains one batch. Returns the
        // average cost as a Scalar.
        if(printInfo){
            /*
            const [p1, x1, t1] = session.evalAll(
                [model.output, x_check, l_check], feedEntries
            );
            console.log("p:", p1.dataSync(), "t:", t1.dataSync());
            */
            for(let i = 0; i <= 100; i++){
                let input = Array1D.new(ds[i].input);
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
        
        const cost = session.train(
            model.cost, feedEntries, batchSize, optimizer, CostReduction.MEAN
        );
        try {
            costVal = await cost.val();
        } catch(e){
            console.log("Error at await", e);
        }
        console.log('last average cost (' + i + '): ' + costVal);
    }

}

function getDeepModel({
    modelType='RNNLSTM',
    nPredictions=outputSize,
    hiddenSize=hiddenSize,
    inputSize=1,
    seqLength=seqLength,
    init_weights=null
}){
    math = ENV.math;
    const graph = new Graph();
    const session = new Session(graph, math);

    let model = null;

    switch (modelType){
        case 'RNNLSTM':
            model = new RNNShared({
                graph: graph, session: session, nPredictions: outputSize,
                hiddenSize: hiddenSize, inputSize: inputSize,
                seqLength: seqLength,
                init_weights: null
            });
            break;
        case 'LSTM':
            // TODO check that seqLength == 1
            model = new RNNShared({
                graph: graph, session: session, nPredictions: outputSize,
                hiddenSize: hiddenSize, inputSize: inputSize,
                seqLength: 1,
                init_weights: null
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

}


// END of export boiler plate
exports.FFN1D = FFN1D;
exports.LSTMCell = LSTMCell;
exports.LSTMCellShared = LSTMCellShared;
exports.RNNShared = RNNShared;
exports.dsToDeepDS = dsToDeepDS;
exports.getDeepModel = getDeepModel;
})(
    typeof exports === 'undefined'?  this['deepmodels']={}: exports
);