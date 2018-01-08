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


function getRandSequence(seq_len=1000, vocab_size=6){
    let seq = []
    for(i = 0; i < seq_len;  i += 1){
        seq.push([Math.floor(Math.random() * vocab_size) + 1])
    }
    return(seq)
}

function getPalindrome(vocab_size=10, seq_len=0){
    if(seq_len == 0){
        seq_len = Math.round(Math.random() * (5 - 1)) + 1;
    }
    const first_half = getRandSequence(seq_len, vocab_size);
    let i = first_half.length;
    let second_half = []
    while(i--){ second_half.push(first_half[i]) };
    const seq = first_half.concat(second_half)
    return(seq);
}

function getNonPalindrome(vocab_size=10, seq_len=0){
    if(seq_len == 0){
        seq_len = Math.round(Math.random() * (5 - 1)) + 1;
    }
    const first_half = getRandSequence(seq_len, vocab_size);
    let second_half = getRandSequence(seq_len, vocab_size);
    let same = true;
    for(let i = 0; i < first_half.length; i++){
        if(second_half[second_half.length - (i + 1)][0] != first_half[i][0]){
            same = false;
        }
    }
    if(same){
        ref_pos = Math.round(Math.random() * (seq_len - 1));
        change_pos = second_half.length - (ref_pos + 1);
        while(second_half[change_pos][0] == first_half[ref_pos][0]){
             second_half[change_pos][0] = Math.floor(
                 Math.random() * vocab_size
             ) + 1;
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

function get_ffn({
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
    datasetSize=1000, flatten_input=true, seq_len=1, vocab_size=2
}){
    let trainSet = []
    for(let i = 0; i < datasetSize / 2; i++){
        let pal = getPalindrome(vocab_size=vocab_size, seq_len=seq_len);
        let non_pal = getNonPalindrome(vocab_size=vocab_size, seq_len=seq_len);
        if(flatten_input){
            pal = [].concat(...pal);
            non_pal = [].concat(...non_pal);
        }
        trainSet.push({input: pal, output: [0,1]});
        trainSet.push({input: non_pal, output: [1,0]});
    }
    return(trainSet);
}

function getMNISTDataset(
    datasetSize=700
){
    const set = mnist.set(datasetSize, 20);
    return(set.training);
}

function getCheckerDataset(datasetSize=1000){

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
    model_type='ffn', hidden_size=[1],
    rate=0.2, iterations=100, error=0.005, train_set_size=1000,
    trainSet=null, input_size=1, output_size=1
}){

    const Architect = synaptic.Architect;
    const Layer = synaptic.Layer;
    const Trainer = synaptic.Trainer;

    let model;

    switch(model_type){
        case "lstm":
            model = new Architect.LSTM(
                input_size, hidden_size, output_size
            );
            break;
        case "perceptron":
            model = new Architect.Perceptron(
                input_size, hidden_size, output_size);
            break;
        case "ffn":
            model = get_ffn({
                "input_size": input_size,
                "hidden_size": hidden_size,
                "output_size": output_size 
            });
            break
    }

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

    return model
}


function draw_palindrome_batches({
    canvas, context, curr, max, ms, model, flatten_input=false,
    seq_len=1, vocab_size=2
}){

    console.log(curr);

    context.clearRect(0, 0, canvas.width, canvas.height);

    to_draw = [];
    for(let i = 0; i < 10; i++){
        pal = getPalindrome(vocab_size=vocab_size, seq_len=seq_len);
        nonPal = getNonPalindrome(vocab_size=vocab_size, seq_len=seq_len);
        if(flatten_input){
            pal = [].concat(...pal);
            nonPal = [].concat(...nonPal);
        }
        to_draw.push([pal, true]);
        to_draw.push([nonPal, false]);
    }


    sub_height = canvas.height / to_draw.length;
    var to_draw_it = to_draw.entries();

    for([idx, [seq, is_pal]] of to_draw_it){
        draw_sequence(
            seq, context, sub_height, sub_height * idx, set_color(is_pal)
        );
        console.log(is_pal, model.activate(seq), seq);
    }
    
    curr = curr + 1;

    if(curr < max){
        console.log("+curr:", curr);
        console.log("+max:", max);
        setTimeout(function(){ draw_palindrome_batches({
            canvas: canvas, context: context, curr: curr, max: max, ms: ms,
            model: model, flatten_input: false, seq_len: 1, vocab_size: 2
        })});
        // sleep(ms).then(draw_batches(curr, max))
    }else{
        console.log("curr:", curr);
        // console.log("max:", max)
    }
}

function ripped_test(){
    
    const set = mnist.set(700, 20);

    const trainingSet = set.training;
    const testSet = set.test;

    const Layer = synaptic.Layer;
    const Network = synaptic.Network;
    const Trainer = synaptic.Trainer;

    const inputLayer = new Layer(784);
    const hiddenLayer = new Layer(100);
    const outputLayer = new Layer(10);

    inputLayer.project(hiddenLayer);
    hiddenLayer.project(outputLayer);

    const myNetwork = new Network({
        input: inputLayer,
        hidden: [hiddenLayer],
        output: outputLayer
    });

    const trainer = new Trainer(myNetwork);
    trainer.train(trainingSet, {
        rate: .2,
        iterations: 20,
        error: .1,
        shuffle: true,
        log: true,
        cost: Trainer.cost.CROSS_ENTROPY
    });
}


window.onload = function(){

    let ret = speed_test(100);
    /*
    let seq = ret.seq
    let synaptic_LSTM = ret.LSTM

    // seq.forEach(function(elt){console.log(elt);}}
    
    synaptic_LSTM.clear()
    synaptic_LSTM.activate(...[seq.slice(0, 10)])

    synaptic_LSTM.clear()
    synaptic_LSTM.activate(...[seq.slice(10, 20)])

    synaptic_LSTM.clear()
    synaptic_LSTM.activate(seq.slice(0, 10))
    synaptic_LSTM.activate(seq.slice(10, 20))
    synaptic_LSTM.activate(seq.slice(0, 10))

    let palindrome = getPalindrome(vocab_size=5)
    console.log(''.concat(...palindrome))

    let non_palindrome = getNonPalindrome(vocab_size=5)
    console.log(''.concat(...non_palindrome))
    */

    let canvas = document.createElement("canvas", id='canvas');
    document.body.appendChild(canvas);
    let height_zoom = 2;
    canvas.width = 1000;
    canvas.height = 600;
    // $("#canvas").css('background-color', 'rgba(158, 167, 184, 0.2)');
    let context = canvas.getContext("2d");

    var model;

    function flatten_setter(){
        if($("#algo").val() == 'lstm'){
            return(false);
        }else{return(true)}
    }

    function data_setter(datasetSize){
        if($("#data").val() == 'palindrome'){
            return(getPalindromeDataset({
                seq_len: $("#seq_len").val(),
                vocab_size: $("#vocab_size").val(),
                flatten_input: flatten_setter()
            }));
        }else if($("#data").val() == 'palindrome'){
            return(getMNISTDataset());
        }else if($("#data").val() == 'checker'){
            return(getCheckerDataset());
        }
    }

    $('#data').change(
        function(){
            currData = $(this).val();
            console.log(currData);
            if(currData == 'mnist'){
                $("#algo").val('ffn')

                $("#n_input").val(784);
                $("#n_hidden").val(100);
                $("#n_output").val(10);

                $("#fset").attr("disabled", true);
                $("#fset").hide();
            }else if(currData == 'palindrome'){
                $("#algo").val('lstm')

                $("#n_input").val(1);
                $("#n_hidden").val(2);
                $("#n_output").val(2);

                $("#fset").attr("disabled", false);
                $("#fset").show();
            }else if(currData == 'checker'){
                $("#algo").val('ffn')

                $("#n_input").val(2);
                $("#n_hidden").val(2);
                $("#n_output").val(2);

                $("#fset").attr("disabled", false);
                $("#fset").show();
            }
        }
    )

    $("#kepler").click(
        () => {
            n_hidden_val = String($("#n_hidden").val())
            n_hidden_array = n_hidden_val.split(',').map((x) => parseInt(x))

            model = learnModel({
                model_type: $("#algo").val(),
                rate: $("#lr").val(),
                batch_num: 1,
                iterations: $("#iter").val(),
                trainSet: data_setter(0),
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
                draw_palindrome_batches({
                    canvas: canvas,
                    context: context,
                    curr: 0,
                    max: 1,
                    ms: 1000,
                    model: model,
                    flatten_input: flatten_setter(),
                    seq_len: $("#seq_len").val(),
                    vocab_size: $("#vocab_size").val()
                })
            }
        }
    );

    $("#takum").click(() => ripped_test());

    /*
    to_draw_it.forEach(
        ([seq, is_pal]) => draw_sequence(
            seq, context, sub_height, 0, set_color(is_pal)
        )
    )
    */

    /*
    draw_sequence(palindrome, context, height_zoom, 0, "#11AE11")
    draw_sequence(non_palindrome, context, height_zoom, 0)
    */

}
