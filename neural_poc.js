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

var get_rand_sequence = function(){
    var seq = []
    for(i = 0; i < 1000;  i += 1){
        seq.push([Math.floor(Math.random() * 6) + 1])
    }
    return(seq)
}

var get_sequence = function(){
    var seq = []
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

var arbitrary_seq_lstm_synaptic= function(){
    var synaptic = require('synaptic');
    var LSTM = new synaptic.Architect.LSTM(10,32,2);
    return(LSTM);
}

var speed_test = function(n_repeat){
    var seq = get_sequence()
    var LSTM = arbitrary_seq_lstm_synaptic()
    var start = new Date().getTime(); // in millis
    for(i = 0; i <= n_repeat; i++){
        LSTM.activate(seq.slice(0, 10))
    }
    var end = new Date().getTime() // in millis
    console.log(
        ''.concat(...[
            'time per predict: ', String(((end - start) / 1000) / n_repeat),
            's'
        ])
    )
    return({
        seq: seq, LSTM: LSTM
    })
}

var ret = speed_test(100)
var seq = ret.seq
var LSTM = ret.LSTM

synaptic_LSTM.clear()
synaptic_LSTM.activate(...[seq.slice(0, 10)])

synaptic_LSTM.clear()
synaptic_LSTM.activate(...[seq.slice(10, 20)])

synaptic_LSTM.clear()
synaptic_LSTM.activate(seq.slice(0, 10))
synaptic_LSTM.activate(seq.slice(10, 20))
synaptic_LSTM.activate(seq.slice(0, 10))
