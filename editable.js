//network specific

//ADJUSTABLES
//linear
let wi = 1;//weight initialization values
let layers = 1;//hidden layers
let hiddensize = 100;//maximum hidden layer size
let inputsize = 64;//max input size
let outputsize = 10;//max output size
let addbias = true;//add bias values to network

//gpt
let traindat = 0.9;//percentage of data used for gpt training
let learningset = 10;//context length
let heads = 3;//heads of gpt network
let querykeydim = 5;//size of smaller dimensional query-key-valuedown space (less then encodesize)
let encodesize = 5;//value up, encoder vector size full representation
let sampleset = 200;//sample data
let smtemperature = 1;//temperature of softmax algorithm
let sureness = 1;//gpt training value
let ffnlayers = 2;//gpt second network layers
let convthresh = 0;//convergence threshold (0-2)

//adamW
let alpha = 0.001;
let b1 = 0.9;
let b2 = 0.999;
let epsilon = 0;

//generative
let newlayer = 0.2;//chance of making new layer in gen
let weightmult = 2;//weights:neurons+biases ratio
let iterations = 50;//models on screen
let timealive = 4;//how long can go without hitting a target (stamina)

//general
let trialspersesh = 100;//various uses, used for averaging results
let learningrate = 1;//velocity of training adjustments
let type = "GELU";//type of activation function (sigmoid,RELU,GELU)
let scale = 1;//normal dist of activation functions

//misc
let textbox;
let lines;
let e = 2.718281828459045;


function screeninfo(totalcost,label,data,netarr) {
	
	//write some stuff for screen
	
}

function preload() {
	//get some file info
}

function setup() {
	
	createCanvas(windowWidth, windowHeight);
	background(0);
	
	//load some data

	//loadshit /something\ ();
	
}

function draw() {
	
	keyPressed();

}

function keyReleased() {
  if (keyCode == SHIFT) {
    keyCode = "";
		return false;
  }
}//press and hold logic

function keyPressed() {
	
	if (keyCode == SHIFT) {
		//call training
	}
	
  return false;

}//train with holding shift

function runexample() {
//load some data
		
	let netarr = runlinear(/*data*/);
	let totalcost = 0;
	for (a = 0; a < netarr.length; a++) {
		if (netarr[a][0] != label) {
			totalcost += netarr[a][1];
			costpertoken[layers][netarr[a][0]] += -1*sureness*netarr[a][1];
		}
		else {
			totalcost += 1-netarr[a][1];
			costpertoken[layers][netarr[a][0]] += correctness*(1-netarr[a][1]);
		}
	}//costcalc
	screeninfo(/*totalcost,label,data,netarr*/)//draw stuff

}
function getNetGuess() {
	
	//load some data for the network from user
	
	//draw stuff on screen
	
}