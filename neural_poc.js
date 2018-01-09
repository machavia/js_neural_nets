// import {Architect} from 'synaptic'; // no need, no sweat...

/*
biblio
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

/* nodejs
var synaptic = require('synaptic')
*/


function getRandSequence(seq_len=1000, vocab_size=6, noise=0){
    let seq = []
    for(i = 0; i < seq_len;  i += 1){
        seq.push([
            Math.floor(Math.random() * vocab_size) + 1 +
            noise * (Math.random() - 0.5)
        ])
    }
    return(seq)
}

function getPalindrome(vocab_size=10, seq_len=0, noise=0){
    if(seq_len == 0){
        seq_len = Math.round(Math.random() * (5 - 1)) + 1;
    }
    const first_half = getRandSequence(seq_len, vocab_size, noise);
    let i = first_half.length;
    let second_half = []
    while(i--){ second_half.push(first_half[i]) };
    const seq = first_half.concat(second_half)
    return(seq);
}

function getNonPalindrome(vocab_size=10, seq_len=0, noise=0){
    if(seq_len == 0){
        seq_len = Math.round(Math.random() * (5 - 1)) + 1;
    }
    const first_half = getRandSequence(seq_len, vocab_size, noise);
    let second_half = getRandSequence(seq_len, vocab_size, noise);
    let same = true;
    for(let i = 0; i < first_half.length; i++){
        if(
            Math.round(second_half[second_half.length - (i + 1)][0]) !=
            Math.round(first_half[i][0])
        ){
            same = false;
        }
    }
    if(same){
        ref_pos = Math.round(Math.random() * (seq_len - 1));
        change_pos = second_half.length - (ref_pos + 1);
        while(
            Math.round(second_half[change_pos][0]) ==
            Math.round(first_half[ref_pos][0])
        ){
             second_half[change_pos][0] = Math.floor(
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

function speed_test(n_repeat){
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
    return({
        seq: seq, LSTM: LSTM
    })
}

function draw_sequence(
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

function get_ffn_brain({
    input_size=1, hidden_size=[2], output_size=2
}){

    var net = new brain.NeuralNetwork({
        hidden_layers: hidden_size
    });
    return(net)
}

function get_lstm_brain({
    input_size=1, hidden_size=[2], output_size=2
}){

    var net = new brain.recurrent.LSTM({
        hidden_layers: hidden_size
    });
    return(net)
}


function get_ffn_synaptic({
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

function set_color(is_pal){
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

function getPalindromeDataset({
    datasetSize=1000, flatten_input=true, seq_len=1, vocab_size=2,
    noise=0
}){
    let trainSet = []
    for(let i = 0; i < datasetSize / 2; i++){
        let pal = getPalindrome(
            vocab_size=vocab_size, seq_len=seq_len, noise=noise
        );
        let non_pal = getNonPalindrome(
            vocab_size=vocab_size, seq_len=seq_len, noise=noise
        );
        if(flatten_input){
            pal = [].concat(...pal);
            non_pal = [].concat(...non_pal);
        }
        trainSet.push({input: pal, output: [0,1]});
        trainSet.push({input: non_pal, output: [1,0]});
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

function learnModel({
    model_type='ffn_synaptic', hidden_size=[1],
    rate=0.2, iterations=100, error=0.005, train_set_size=1000,
    trainSet=null, input_size=1, output_size=1
}){

    const Architect = synaptic.Architect;
    const Layer = synaptic.Layer;
    const Trainer = synaptic.Trainer;

    let model;

    switch(model_type){
        case "lstm_synaptic":
            model = new Architect.LSTM(
                input_size, hidden_size, output_size
            );
            break;
        case "perceptron_synaptic":
            model = new Architect.Perceptron(
                input_size, hidden_size, output_size
            );
            break;
        case "ffn_synaptic":
            model = get_ffn_synaptic({
                "input_size": input_size,
                "hidden_size": hidden_size,
                "output_size": output_size 
            });
            break
        case 'ffn_brain':
            model = get_ffn_brain({
                "hidden_size": hidden_size,
            });
            break;
        case 'lstm_brain':
            model = get_lstm_brain({
                "hidden_size": hidden_size,
            });
            break;
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
    }

    return model
}


function test_batch({
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
        is_pal = ! Boolean(output[0]);
        draw_sequence(
            seq, context, sub_height, sub_height * idx, set_color(is_pal)
        );

        if(model_type.match('.*_synaptic')){
            pred = model.activate(seq)
        }else if(model_type.match('.*_brain')){
            pred = model.run(seq)
        }

        console.log(is_pal, pred, seq);
    }
    
    curr = curr + 1;

    if(curr < max){
        console.log("+curr:", curr);
        console.log("+max:", max);
        setTimeout(
            () => test_batch({
            canvas: canvas, context: context, curr: curr, max: max, ms: ms,
            model: model, flatten_input: false, seq_len: 1, vocab_size: 2
            })
        );
        // sleep(ms).then(draw_batches(curr, max))
    }else{
        console.log("curr:", curr);
        // console.log("max:", max)
    }
}

var debug = {
    trainSet: null,
    testSet: null,
    model: null
}

window.onload = function(){

    let ret = speed_test(100);

    let canvas = document.createElement("canvas", id='canvas');
    document.body.appendChild(canvas);
    let height_zoom = 2;
    canvas.width = 1000;
    canvas.height = 600;
    // $("#canvas").css('background-color', 'rgba(158, 167, 184, 0.2)');
    let context = canvas.getContext("2d");

    var model;

    function flatten_setter(){
        if(
            $("#algo").val() == 'lstm_synaptic' ||
            $("#algo").val() == 'lstm_brain'
        ){
            return(false);
        }else{return(true)}
    }

    function data_setter(type='train', flatten=null){
        if(flatten == null){
            flatten = flatten_setter()
        }
        let datasetSize
        if(type == 'train'){
            datasetSize = $("#train_size").val()
        }else{
            datasetSize = $("#test_size").val()
        }
        datasetSize = parseInt(datasetSize)
        if($("#data").val() == 'palindrome'){
            return(getPalindromeDataset({
                datasetSize: datasetSize,
                seq_len: $("#seq_len").val(),
                vocab_size: $("#vocab_size").val(),
                flatten_input: flatten, 
                noise: $("#noise").val()
            }));
        }else if($("#data").val() == 'mnist'){
            return(getMNISTDataset({datasetSize: datasetSize}));
        }else if($("#data").val() == 'checker'){
            return(getCheckerDataset({datasetSize: datasetSize}));
        }
    }

    $('#data').change(
        function(){
            currData = $(this).val();
            console.log(currData);
            if(currData == 'mnist'){
                $("#algo").val('ffn_synaptic')

                $("#n_input").val(784);
                $("#n_hidden").val(100);
                $("#n_output").val(10);

                $("#pal_fset").attr("disabled", true);
                $("#pal_fset").hide();
            }else if(currData == 'palindrome'){
                $("#algo").val('lstm_synaptic')

                $("#n_input").val(1);
                $("#n_hidden").val(2);
                $("#n_output").val(2);

                $("#pal_fset").attr("disabled", false);
                $("#pal_fset").show();
            }else if(currData == 'checker'){
                $("#algo").val('ffn_synaptic')

                $("#n_input").val(2);
                $("#n_hidden").val(2);
                $("#n_output").val(2);

                $("#pal_fset").attr("disabled", false);
                $("#pal_fset").hide();
            }
        }
    )

    $("#kepler").click(
        () => {
            trainSet = data_setter("train");
            debug.trainSet = trainSet
            n_hidden_val = String($("#n_hidden").val())
            n_hidden_array = n_hidden_val.split(',').map((x) => parseInt(x))

            model = learnModel({
                model_type: $("#algo").val(),
                rate: $("#lr").val(),
                batch_num: 1,
                iterations: $("#iter").val(),
                trainSet: trainSet,
                input_size: $("#n_input").val(),
                hidden_size: n_hidden_array,
                output_size: $("#n_output").val(),
            });
            console.log(model);
        }
    );

    $("#ptolemee").click(
        () => {
            if(model == null){
                console.log('train first !!!')
            }else{
                testSet = data_setter('test')
                debug.testSet = testSet
                test_batch({
                    canvas: canvas,
                    context: context,
                    curr: 0,
                    max: 1,
                    ms: 1000,
                    model: model,
                    testSet: testSet,
                    model_type: $("#algo").val()
                })
            }
        }
    );

}
