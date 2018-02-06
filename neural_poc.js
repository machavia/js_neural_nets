getPalindromeDataset = utils.getPalindromeDataset;
drawSequence = utils.drawSequence;
getPalindromeDataset = utils.getPalindromeDataset;
getFFNBrain = utils.getFFNBrain;
getLstmBrain = utils.getLstmBrain;
learnModel = utils.learnModel;
getFFNSynaptic = utils.getFFNSynaptic;
speedTest = utils.speedTest;
testBatch = utils.testBatch;
debug = utils.debug;
debugDeep = deepmodels.debug;

window.onload = function(){

    // new SharedWorker('worker.js');

    let ret = speedTest(100);

    let canvas = document.createElement("canvas", id='canvas');
    document.body.appendChild(canvas);
    let height_zoom = 2;
    canvas.width = 1000;
    canvas.height = 600;
    // $("#canvas").css('background-color', 'rgba(158, 167, 184, 0.2)');
    let context = canvas.getContext("2d");

    var model;

    function flattenSetter(){
        if(
            $("#algo").val() == 'lstm_synaptic' ||
            $("#algo").val() == 'lstm_brain'
        ){
            return(false);
        }else{return(true)}
    };

    function dataSetter(type='train', flatten=null){
        if(flatten == null){
            flatten = flattenSetter();
        }
        let datasetSize
        if(type == 'train'){
            datasetSize = $("#train_size").val();
        }else{
            datasetSize = $("#test_size").val();
        }
        datasetSize = parseInt(datasetSize)
        if($("#data").val() == 'palindrome'){
            let ds = getPalindromeDataset({
                datasetSize: datasetSize,
                halfSeqLen: parseInt($("#half_len").val()),
                vocab_size: parseInt($("#vocab_size").val()),
                flatten: flatten, 
                noise: parseFloat($("#noise").val()),
                dim: parseInt($("#dim").val())
            });
            return(ds);
        }else if($("#data").val() == 'mnist'){
            return(getMNISTDataset({datasetSize: datasetSize}));
        }else if($("#data").val() == 'checker'){
            return(getCheckerDataset({datasetSize: datasetSize}));
        }
    };

    $('#data').change(
        () => {
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
    );

    $("#kepler").click(
        async () => {
            trainSet = dataSetter("train");
            debug.trainSet = trainSet;
            n_hidden_val = String($("#n_hidden").val());
            n_hidden_array = n_hidden_val.split(',').map((x) => parseInt(x));

            const inneriIterations = parseInt($("#iter").val());
            const outerIterations= 1;

            for (let i = 0; i < outerIterations; i++){
                model = await learnModel({
                    model_type: $("#algo").val(),
                    rate: parseFloat($("#lr").val()),
                    batchSize: parseInt($("#batch_size").val()),
                    iterations: inneriIterations,
                    trainSet: trainSet,
                    input_size: parseInt($("#n_input").val()),
                    hidden_size: n_hidden_array,
                    output_size: parseInt($("#n_output").val()),
                    seqLength: parseInt($("#seq_length").val()),
                    optimizer: $("#optimizer").val(),
                    optimizerByBatch:
                        Boolean(parseInt($("#optimizerByBatch").val())),
                    modelByBatch: Boolean(parseInt($("#modelByBatch").val()))
                });
            }
        }
    );

    $("#ptolemee").click(
        () => {
            if(model == null){
                console.log('train first !!!')
            }else{
                testSet = dataSetter('test');
                debug.testSet = testSet;
                testBatch({
                    canvas: canvas,
                    context: context,
                    curr: 0,
                    max: 1,
                    ms: 1000,
                    model: model,
                    testSet: testSet,
                    model_type: $("#algo").val()
                });
            }
        }
    );

}
