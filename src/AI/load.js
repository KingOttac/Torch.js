//network consts

//gpt
let costpertoken = [];//assigned by neuron number, represents derivative of cost function
let convertedlines = [];//giant array for dataset
let encoders = [];//vectors to encode tokens
let decoders = [];//vectors to decode tokens
let key = [];
let query = [];
let value = [];
let returns = [];
let tokens = [];
//gpt train
let mt = [];
let vt = [];
let tsp = [];
let correctcheck = [];

//linear
let neuronstore = [];
let weights = [];
let biases = [];
let costarr = [];

//generative
let scores = [];//generation scores
let currentbest = 0;

function loadlinear() {
	
	weights = maketensor(3,[layers,hiddensize,hiddensize],0,true,-1*wi,wi);
	weights[0] = maketensor(2,[encodesize,hiddensize],0,true,-1*wi,wi);
	weights[layers] = maketensor(2,[hiddensize,outputsize],0,true,-1*wi,wi);
	biases = maketensor(2,[layers+1,hiddensize],0,true,-1*wi,wi);
	biases[layers] = maketensor(1,[outputsize],0,true,-1*wi,wi);
	
}

function loadGen() {
	
	scores = maketensor(1,[iterations],0);//updated in draw
	neuronstore = maketensor(2,[iterations,2],[]);
	for (a = 0; a < iterations; a++) {
		neuronstore[a][0] = [];
		neuronstore[a][1] = [];
		for (b = 0; b < inputsize; b++) {
			neuronstore[a][0][b] = {
				index:[0,b],
				value:0
			}
		}
		for (b = 0; b < outputsize; b++) {
			neuronstore[a][1][b] = {
				index:[1,b],
				weights:[],
				bias:0,
				value:0
			}
		}
	}//input and output layer adjust
	trainGen();
	
}

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
	weights = maketensor(1,[layers],shapenet([inputsize,hiddensize,outputsize],false,2,ffnlayers,0,true,-wi,wi));
	biases = maketensor(1,[layers],shapenet([hiddensize,hiddensize,outputsize],false,1,ffnlayers,0,true,-wi,wi));
	
	//unchanging
	returns = maketensor(1,[learningset],untoken("\n"));
	
	//training
	function filler(fv) {
		return [
			maketensor(4,[layers,heads,encodesize,querykeydim],fv),
			maketensor(4,[layers,heads,encodesize,querykeydim],fv),
			maketensor(4,[layers,heads,encodesize,querykeydim],fv),
			maketensor(4,[layers,heads,querykeydim,encodesize],fv),
			maketensor(1,[layers],shapenet([inputsize,hiddensize,outputsize],false,2,ffnlayers,fv)),
			maketensor(1,[layers],shapenet([hiddensize,hiddensize,outputsize],false,1,ffnlayers,fv)),
			maketensor(2,[tokens.length,encodesize],fv)
		];
	}
	mt = filler(0);
	vt = filler(0);
	tsp = 0;
	params = [key,query,valuedown,valueup,weights,biases,encoders];

}

function loadlinear() {

	weights = maketensor(3,[layers,hiddensize,hiddensize],0,true,-wi,wi);
	weights[0] = maketensor(2,[encodesize,hiddensize],0,true,-wi,wi);
	weights[layers] = maketensor(2,[hiddensize,outputsize],0,true,-wi,wi);
	biases = maketensor(2,[layers+1,hiddensize],0,true,-wi,wi);
	biases[layers] = maketensor(1,[outputsize],0,true,-wi,wi);

}

function tokenizer(type) {
	
	for (a = 0; a < sampleset; a++) {
		
		let listarr = [];
		if (type == "space") {
			listarr = lines[a] + " \n";
			listarr = split(listarr," ");
		}
		else if (type == "char") {
			listarr = lines[a] + "\n";
			listarr = split(listarr,"");
		}
		for (b = 0; b < listarr.length; b++) {
			if (listarr[b] != "\n" && type == "space") {
				listarr[b] += " ";
			}
			convertedlines[convertedlines.length] = listarr[b];
			if (untoken(listarr[b]) == -1) {
				tokens[tokens.length] = listarr[b];
			}
		}
		
	}
	
	for (a = 0; a < convertedlines.length; a++) {
		convertedlines[a] = untoken(convertedlines[a])
	}//convert everything into numbers
	
}//converts one data file (from preload) into tokens in the final array
