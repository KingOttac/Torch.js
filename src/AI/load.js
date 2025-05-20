//gpt
let convertedlines = [];//giant array for dataset
let encoders = [];//vectors to encode tokens
let key = [];
let query = [];
let value = [];
let tokens = [];
let params = [];

//linear
let costpertoken = [];//assigned by neuron number, represents derivative of cost function
let neuronstore = [];
let weights = [];
let biases = [];
let costarr = [];

//generative
let scores = [];//generation scores
let currentbest = 0;
let totalbest = false;

function loadGPT() {

	//attention
	key = maketensor(4,[layers,heads,encodesize,querykeydim],0,true,-wi,wi);
	query = maketensor(4,[layers,heads,encodesize,querykeydim],0,true,-wi,wi);
	valuedown = maketensor(4,[layers,heads,encodesize,querykeydim],0,true,-wi,wi);
	valueup = maketensor(4,[layers,heads,querykeydim,encodesize],0,true,-wi,wi);
	encoders = maketensor(2,[tokens.length,encodesize],0,true,-wi,wi);
	
	//ffn
	inputsize = encodesize;
	hiddensize = 4*encodesize;
	outputsize = encodesize;
	let shapenetcall = function(parr,inputs) {
		return shapenet([inputsize,hiddensize,outputsize],false,inputs[0],ffnlayers,0,true,-wi,wi);
	}
	weights = maketensor(1,[layers],shapenetcall,[2]);
	biases = maketensor(1,[layers],shapenetcall,[1]);

	params = [key,query,valuedown,valueup,weights,biases,encoders];
	
}

function loadlinear() {

	weights = shapenet([inputsize,hiddensize,outputsize],false,2,layers,0,true,-wi,wi);//load weights
	biases = shapenet([hiddensize,hiddensize,outputsize],false,1,layers,0,true,-wi,wi);//load biases

}

function loadGen() {
	
	scores = maketensor(1,[iterations],0);//updated in draw
	let makenetwork = function(parr,inputs) {
		return {
			neurons:[tensor(0,[inputsize]),[0]],
			weights:shapenet([inputsize,0,1],false,2,0,0),
			biases:shapenet([inputsize,0,1],false,1,0,0)
		}
	}
	neuronstore = maketensor(1,[iterations],makenetwork);
	trainGen();
		
}
