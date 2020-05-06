
 const model = tf.sequential();

	//FUNCTION FOR TRAINING

        async function doTraining(EPOCHS, XS, YS){
        	document.getElementById("train_progress").max = EPOCHS;
            const history = await model.fit(XS, YS, {
            	epochs: EPOCHS, callbacks: {
            		onEpochEnd: async(epoch, logs) => {
            			console.log("Epochs:" + epoch + "Loss: " + logs.loss + ' Accuracy: ', parseFloat(logs.acc));
				document.getElementById("progress").innerHTML = "Epochs: "+(epoch+1)+" / "+EPOCHS;
				document.getElementById("current_loss").innerHTML = "Loss: "+logs.loss;
				document.getElementById("train_progress").value = epoch+1;

            		}
            	}
            }
            );
            alert("Training is Complete. You can now predict Y");
            firstRun = false;
        }



    //FUNCTION FOR INITIALIZING MODEL



        async function initializeModel(LAYER1NEURON, LAYER2NEURON, LOSS, OPTIMIZER){
        model.add(tf.layers.dense({units: LAYER1NEURON, inputShape: [1]}));
        model.add(tf.layers.dense({units: LAYER2NEURON}));
        model.add(tf.layers.dense({units: 1}));
        model.compile({loss: LOSS, optimizer: OPTIMIZER,  metrics: 'accuracy'});
    }


var train = document.getElementById("train");
var predict = document.getElementById("predict");
var flag = true;
var firstRun = true;

train.addEventListener("click", async function() {

	//GETTING USERS INPUTS

	var x = await document.getElementsByClassName("x");
	var y = await document.getElementsByClassName("y");
	var layer_1_neuron =parseInt(document.getElementById("layer1").value);
	var layer_2_neuron =parseInt(document.getElementById("layer2").value);
	var no_of_epochs = parseInt(document.getElementById("epoch").value);
	var learning_rate = parseInt(document.getElementById("lr").value);
	var loss_function = parseInt(document.querySelector('#losses').value);
	var optimizer_select = parseInt(document.querySelector('#optimizers').value);

	//CHECKING THE INPUTS AND SETTING DEFAULTS


	for(var i = 0; i<6; i++)if(!(((parseFloat(x[i].value)) && (parseFloat(y[i].value))) || ((parseFloat(x[i].value)==0) || (parseFloat(y[i].value)==0))))
	 { flag = false; alert("Please fill in all the values before training!"+"\n"+"All the values should be numbers.");  break;}
	 else{flag = true}

	if(!layer_1_neuron)layer_1_neuron=1;
	if(!layer_2_neuron)layer_2_neuron=1;
	if(!no_of_epochs)no_of_epochs=500;
	if(!learning_rate)learning_rate=0.001;

	var lr = learning_rate;
	var optimizers;
	var losses;

	switch(optimizer_select){

		case 0: optimizers = tf.train.sgd(lr); break;
		case 1: optimizers = tf.train.momentum(lr, 0.9); break;
		case 2: optimizers = tf.train.adagrad(lr); break;
		case 3: optimizers = tf.train.adadelta(lr); break;
		case 4: optimizers = tf.train.adam(lr); break;
		case 5: optimizers = tf.train.adamax(lr); break;
		case 6: optimizers = tf.train.rmsprop(lr); break;
		default: optimizers = tf.train.sgd(lr); break;

	}

	switch(loss_function){

		case 0: losses = tf.losses.meanSquaredError; break;
		case 1: losses = tf.losses.absoluteDifference; break;
		case 2: losses = tf.losses.computeWeightedLoss; break;
		case 3: losses = tf.losses.cosineDistance; break;
		case 4: losses = tf.losses.hingeLoss; break;
		case 5: losses = tf.losses.huberLoss; break;
		case 6: losses = tf.losses.logLoss; break;

		default: losses = tf.losses.meanSquaredError; break;

	}


var xs = await tf.tensor2d([parseFloat(x[0].value),parseFloat(x[1].value),parseFloat(x[2].value),parseFloat(x[3].value),parseFloat(x[4].value),parseFloat(x[5].value)],[6,1]);
var ys = await tf.tensor2d([parseFloat(y[0].value), parseFloat(y[1].value),parseFloat(y[2].value),parseFloat(y[3].value),parseFloat(y[4].value),parseFloat(y[5].value)],[6,1]);

		if(firstRun)initializeModel(layer_1_neuron, layer_2_neuron, losses, optimizers);



		if(flag)doTraining(no_of_epochs, xs, ys);



        			});



		predict.addEventListener("click", async function(){


			var test_value = document.getElementById("test_value");
			test_value = test_value.value;
			var a = model.predict(tf.tensor2d([parseFloat(test_value)], [1, 1]));
			console.log(test_value);
			alert(a.dataSync());



		});