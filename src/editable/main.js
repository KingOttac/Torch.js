//network specific

//  /X\ indicates it changes function name- need docs

//ADJUSTABLES
//linear
let inputsize = 0;//max input size
let hiddensize = 0;//maximum hidden layer size
let outputsize = 0;//max output size
let addbias = false;//add bias values to network

//gpt
//dataset
let traindat = 0.9;//0-1 section of data used for gpt training
let learningset = 0;//context length
let sampleset = 0;//total number of data samples
let lineset = 0;//lines of dataset
//network
let heads = 0;//heads of gpt network
let querykeydim = 0;//size of smaller dimensional query-key-valuedown space (less then encodesize)
let encodesize = 0;//value up, encoder vector size full representation
let smtemperature = 1;//temperature of softmax algorithm
let sureness = 0;//gpt training value
let ffnlayers = 0;//gpt second network layers
let convthresh = 0;//convergence threshold (0-2)

//adamW
let alpha = 0.001;
let b1 = 0.9;
let b2 = 0.999;
let epsilon = 0.000001;

//generative
let iterations = 10;//models on screen
let inputbias = 1;//final input bias on input arr
let topperc = 0.1;//fraction that dont evolve
let perc = [0.9,0.05,0.05];

//general
let wi = 1;//weight initialization values
let layers = 0;//hidden layers
let trialspersesh = 0;//various uses, used for averaging results
let learningrate = 0.5;//velocity of training adjustments
let type = "sigmoid";//type of activation function (sigmoid,RELU,GELU)
let scale = 1;//normal dist of activation functions

//misc
let textbox;
let lines;
let pi = 3.14159265358979;
let INF = 1e+308;

function preload() {
	//get some file info
}

function setup() {
	
	createCanvas(windowWidth, windowHeight);
	background(0);
	textAlign(CENTER);
	fill(255);
	textSize(30)
	text("Welcome to Torch.js." +
		 "\nThis is currently being loaded from the default setup() function." +
		 "\nPlease edit main.js to implement code",windowWidth/2,windowHeight/2.3);
	
	//load some data

	//loadshit/modeltype\();
	
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
		//train/modeltype\();
	}
	
  return false;

}//train with holding shift

function screeninfo(totalcost,label,data,netarr) {
	
	//write some stuff for screen
	
}

function runexample(data,disp==true) {
//load some data
		
	//let netarr = run/model\(data);
	/*let totalcost = 0;
	for (let a = 0; a < netarr.length; a++) {
		if (netarr[a][0] != label) {
			totalcost += netarr[a][1];
			costpertoken[layers][netarr[a][0]] += -1*sureness*netarr[a][1];
		}
		else {
			totalcost += 1-netarr[a][1];
			costpertoken[layers][netarr[a][0]] += correctness*(1-netarr[a][1]);
		}
	}//costcalc*/
	/*
 	if (disp == true) {
 		screeninfo(totalcost,label,data,netarr)
   	}*/

	return /*variable*/

}

function getinput() {
	
	let trainingIs = /*access dataset here*/
	let label = /*load label for dataset example*/
	return [trainingIs,label];//returned to the training function
	
}//load a single example

function getNetGuess() {
	
	//load some data for the network from user
	
	//draw stuff on screen
	
}
