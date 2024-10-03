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

function adamW(p0,p1,p2,p3,pv) {
		
	function dimen(assign,arr,pv,val) {
		if (assign == true) {
			if (dsties[pv] == 2) {
				arr[p0][p1] = val;
			}//encoder changes
			else if (dsties[pv] == 3) {
				arr[p0][p1][p2] = val;
			}//bias changes
			else if (dsties[pv] == 4) {
				arr[p0][p1][p2][p3] = val;
			}//key, query, valdown, valup, weights
		}
		else {
			if (dsties[pv] == 2) {
				return arr[p0][p1];
			}//encoder changes
			else if (dsties[pv] == 3) {
				return arr[p0][p1][p2];
			}//bias changes
			else if (dsties[pv] == 4) {
				return arr[p0][p1][p2][p3];
			}//key, query, valdown, valup, weights
		}
	}//different dimensional arrays- assign: set to or return

	//move in from array
	let mtin = dimen(false,mt[pv],pv);
	let vtin = dimen(false,vt[pv],pv);
	let tspin = dimen(false,tsp[pv],pv);

	//calculate vec adjust
	tspin++;
	let randomspot = rr(learningset,convertedlines.length);
	let rereturn = runexample(randomspot);
	let gtin = cc[randomspot-learningset]-rereturn;
	mtin = mtin + gtin;//get first vec change
	vtin = vtin + pow(gtin,2);//get second vec change

	//move back changed values
	dimen(true,mt[pv],pv,mtin)
	dimen(true,vt[pv],pv,vtin)
	dimen(true,tsp[pv],pv,tspin)
	cc[randomspot-learningset] = rereturn;

	//send a value to param
	let rturn = -alpha*mtin/sqrt(vtin);
	if (isNaN(rturn) == true) {
		return 0;
	}
	else {
		return rturn;
	}

}

//network consts

function loadshitlinear() {
	
	weights = maketensor(3,[layers,hiddensize,hiddensize],0,true,-1*wi,wi);
	weights[0] = maketensor(2,[encodesize,hiddensize],0,true,-1*wi,wi);
	weights[layers] = maketensor(2,[hiddensize,outputsize],0,true,-1*wi,wi);
	biases = maketensor(2,[layers+1,hiddensize],0,true,-1*wi,wi);
	biases[layers] = maketensor(1,[outputsize],0,true,-1*wi,wi);
	
}

function loadshitGen() {
	
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

function loadshitGPT() {

	//attention
	key = maketensor(4,[layers,heads,encodesize,querykeydim],0,true,0,wi);
	query = maketensor(4,[layers,heads,encodesize,querykeydim],0,true,0,wi);
	valuedown = maketensor(4,[layers,heads,encodesize,querykeydim],0,true,0,wi);
	valueup = maketensor(4,[layers,heads,querykeydim,encodesize],0,true,0,wi);
	
	//unchanging
	returns = maketensor(1,[learningset],untoken("\n"));
	encoders = maketensor(2,[tokens.length,encodesize],0,true,0,wi);
	
	//ffn
	let ffnfill = 
	shapenet([[heads*encodesize,heads*encodesize],[heads*encodesize,heads*encodesize],[heads*encodesize,encodesize]],
					 false,2,ffnlayers-2,0,true,0,wi);
	weights = maketensor(1,[layers],ffnfill);
	let ffnbias = 
	shapenet([heads*encodesize,heads*encodesize,encodesize],false,1,ffnlayers-2,0,true,0,wi);
	biases = maketensor(1,[layers],ffnbias);
	
	//training
	let axm = max(heads,ffnlayers);
	mt = maketensor(5,[6,layers,axm,heads*encodesize,heads*encodesize],0);
	vt = maketensor(5,[6,layers,axm,heads*encodesize,heads*encodesize],0);
	tsp = maketensor(5,[6,layers,axm,heads*encodesize,heads*encodesize],0);
	correctcheck = maketensor(5,[6,layers,axm,heads*encodesize,heads*encodesize],[4,2]);

}

function loadshitlinear() {

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


function runGen(input,sorted,iid,outputarr,allnetworks) {
	
	for (gg = 0; gg < inputsize; gg++) {
		if (activatein == true) {
			allnetworks[iid][0][gg].value = activate([input[gg]])[0];
		}
		else {
			allnetworks[iid][0][gg].value = input[gg];
		}
	}//prepare input
	for (gg = 1; gg < allnetworks[iid].length; gg++) {//layers
		for (gg1 = 0; gg1 < allnetworks[iid][gg].length; gg1++) {//neurons in layer
			allnetworks[iid][gg][gg1].value = 0;
		}
	}//zero out rest of network
	for (gg = 1; gg < allnetworks[iid].length; gg++) {//layers
		for (gg1 = 0; gg1 < allnetworks[iid][gg].length; gg1++) {//neurons in layer
			for (gg2 = 0; gg2 < allnetworks[iid][gg][gg1].weights.length; gg2++) {//layers of weights
				if (allnetworks[iid][gg][gg1].weights[gg2] === undefined) {
					continue;
				}
				for (gg3 = 0; gg3 < allnetworks[iid][gg][gg1].weights[gg2].length; gg3++) {//neuron in layer of weights
					if (allnetworks[iid][gg][gg1].weights[gg2][gg3] === undefined) {
						continue;
					}
					allnetworks[iid][gg][gg1].value += 
						allnetworks[iid][gg][gg1].weights[gg2][gg3]*
						allnetworks[iid][gg2][gg3].value;
				}
			}
			if (addbias == true) {
				allnetworks[iid][gg][gg1].value += allnetworks[iid][gg][gg1].bias;
			}
			allnetworks[iid][gg][gg1].value = activate([neuronstore[iid][gg][gg1].value])[0];
		}
	}//apply weights and biases
	
	let returnarr = [];
	for (gg = 0; gg < outputsize; gg++) {
		returnarr[gg] = allnetworks[iid][allnetworks[iid].length-1][gg].value;
	}
	if (sorted == true) {
		return Bsort(returnarr,outputarr,false);
	}
	else {
		return returnarr;
	}
	
}

function runGPT(input) {

	//multihead self attention
	let last = [];
	for (ll = 0; ll < input.length; ll++) {
		last[ll] = CA(encoders[input[ll]])
	}//sets up encoders of last
	neuronstore = maketensor(3,[layers],0);//init
	for (ll = 0; ll < layers; ll++) {
		//multihead self attention
		let fval = maketensor(3,[heads,learningset,encodesize],0);
		for (hh = 0; hh < heads; hh++) {
			let normtensor = tril(maketensor(2,[learningset,learningset],0),-INF);//init- [query][key]
			for (b = 0; b < learningset; b++) {
				let qdot = matrixmult([CA(last[b])],CA(query[ll][hh]));//gets query vec with matrix
				for (c = 0; c < learningset; c++) {
					if (normtensor[b][c] == -INF) {
						break;
					}//mask rest of row
					let encoded = add2d([CA(last[c])],[maketensor(1,[encodesize],positioners(c+1))]);//encode position into word vec
					let kdot = matrixmult(encoded,CA(key[ll][hh]));//gets corresponding key vec with matrix
					normtensor[b][c] = matrixmult(qdot,transpose(kdot))[0][0]/sqrt(querykeydim);//find vec association and scale (basically dot product)
				}
			}//dot all queries with all input based keys
			for (b = 0; b < learningset; b++) {
				normtensor[b] = softmax(normtensor[b]);//softmax vector associations
			}//softmax normtensor rows
			for (a = 0; a < learningset; a++) {
				//get vector shift with scaled value vec
				let curval = matrixmult([CA(last[a])],CA(valuedown[ll][hh]));//valuedown*key encode gives (down: [qkdim])
				curval = matrixmult(curval,CA(valueup[ll][hh]))[0];//valueup*curval gives (back up to [encode])
				for (b = 0; b < learningset; b++) {
					fval[hh][a] = add2d([CA(fval[hh][a])],mult2d([CA(curval)],maketensor(2,[1,encodesize],normtensor[a][b])))[0];
				}
			}//update desired changes to last
		}//perform attention
		for (hh = 0; hh < heads; hh++) {
			last = add2d(last,fval[hh]);
		}//edit last
		
		//feed forward network
		for (hh = 0; hh < learningset; hh++) {
			last[hh] = normalize(last[hh],wi);
			let linout = runlinear(CA(last[hh]),ffnlayers+1,false,weights[ll],biases[ll])//gets neuron arrangement[1] and new output flow values[0]
			last[hh] = normalize(add2d([last[hh]],[linout[0]])[0],wi);//add and normalize
		}
	}//multilayer gpt oh yeah
	
	return matrixmult([last[last.length-1]],transpose(encoders))[0];//[0] is just to lower dimension
	
}//takes in previous tokens as numbers

function runlinear(input,qlayers,sorted,allweights,allbiases) {
	
	function linear(ARR,weightsarr,biasesarr) {

		let returnarr = matrixmult([ARR],weightsarr)[0];
		if (addbias == true) {
			returnarr = add2d([returnarr],[biasesarr])[0];
		}

		return returnarr;

	}//takes in 1d array and returns one transform with weights from layer + bias

	let nsra = [];
	nsra[0] = input;
	let ra = input;
	for (a = 0; a < qlayers; a++) {
		nsra[a+1] = linear(ra,allweights[a],allbiases[a]);
		ra = activate(nsra[a+1])
	}
	if (sorted == true) {
		return [Bsort(ra,outputarr),nsra];
	}
	else {
		return [ra,nsra];
	}

}//takes in parameters for layers (corr to weights), if sorted, input 
//returns an array of [output , each layer unactivated arr]

function trainGen() {
	
	//bests
	let tsbests = Bsort(CA(scores),maketensor(1,[iterations],0,false,0,0,false,true),false,true);
	currentbest = tsbests[0][0];
	
	//evolve networks (copies to second to take only top half)
	let final = [];
	let divline = round(iterations/2*(1-perevolve));//where to switch from keep to change
	let loopmover = -1;//used to only keep best networks
	for (gg1 = 0; gg1 < divline; gg1++) {
		if (gg1 != 0 && tsbests[gg1][1] != tsbests[gg1-1][1]) {
			loopmover = 0;
		}
		else {
			loopmover++;
		}
		final[gg1] = neuronstore[tsbests[loopmover][0]];
	}//kept networks first section
	for (gg1 = divline; gg1 < round(iterations/2); gg1++) {
		for (ggr = rr(0,learningrate); ggr > 0; ggr--) {
			if (random(0,1) < newlayer && neuronstore[gg1].length < layers+2) {
				let leng = neuronstore[gg1].length;
				neuronstore[gg1][leng] = CA(neuronstore[gg1][leng-1],true);//shift output to right
				for (gg2 = 0; gg2 < outputsize; gg2++) {
					neuronstore[gg1][leng][gg2].index[0]++;
				}//shift output indexes
				neuronstore[gg1][leng-1] = [{
					index:[leng-1,0],
					weights:[],
					bias:0,
					value:0
				}];//insert new neuron list
			}//make new layer
			else if (neuronstore[gg1].length > 2) {
				let randlay = rr(1,neuronstore[gg1].length-1);
				let leng = neuronstore[gg1][randlay].length;
				if (leng < hiddensize) {
							neuronstore[gg1][randlay][leng] = {
							index:[randlay,leng],
							weights:[],
							bias:0,
							value:0
						};
					}
			}//add neuron to existing layer
		}//new neurons
		for (ggr = rr(0,weightmult*learningrate); ggr > 0; ggr--) {
			let randlay = rr(1,neuronstore[gg1].length);//excludes first lay- second
			let randwei = rr(0,neuronstore[gg1][randlay].length);//get rand in lay- second
			let randconlay = rr(0,randlay);//layer of index- first
			let randconwei = rr(0,neuronstore[gg1][randconlay].length);//random neuron in- first
			if (neuronstore[gg1][randlay][randwei].weights[randconlay] === undefined) {
				neuronstore[gg1][randlay][randwei].weights[randconlay] = [];
			}
			neuronstore[gg1][randlay][randwei].weights[randconlay][randconwei] = random(-wi,wi);//set weight
		}//new weights
		for (ggr = rr(0,learningrate); ggr > 0; ggr--) {
			let randlay = rr(1,neuronstore[gg1].length);//excludes first lay- second
			let randwei = rr(0,neuronstore[gg1][randlay].length);//get rand in lay- second
			neuronstore[gg1][randlay][randwei].bias += random(-wi,wi);//set bias
		}//new biases
		final[gg1] = neuronstore[gg1];
	}//random traits first section
	for (gg1 = round(iterations/2); gg1 < iterations; gg1++) {
		final[gg1] = final[gg1-round(iterations/2)];
	}//copy both to second section
	neuronstore = final;
	
	scores = maketensor(1,[iterations],0);
	
}

function trainGPT() {
		
	//vector case
	let llt = rr(0,layers);
	let hht = rr(0,heads);
	let bt = rr(0,encodesize);
	let ct = rr(0,querykeydim);
	
	//weight case
	let hhtw = rr(0,ffnlayers);
	let btw = rr(0,weights[llt][hhtw].length);
	let ctw = rr(0,weights[llt][hhtw][btw].length);
	
	//bias case
	let btb = rr(0,biases[llt][hhtw].length);
	
	//encode case
	let llte = rr(0,tokens.length);
	let hhte = rr(0,encodesize);
	
	//adjust random parameter
	switch(rr(0,7)) {
		case 0:
			key[llt][hht][bt][ct] += adamW(llt,hht,bt,ct,0);
		break;
		case 1:
			query[llt][hht][bt][ct] += adamW(llt,hht,bt,ct,0);
		break;
		case 2:
			valuedown[llt][hht][bt][ct] += adamW(llt,hht,bt,ct,0);
		break;
		case 3:
			valueup[llt][hht][ct][bt] += adamW(llt,hht,ct,bt,0);
		break;
		case 4:
			weights[llt][hhtw][btw][ctw] += adamW(llt,hhtw,btw,ctw,0);
		break;
		case 5:
			biases[llt][hhtw][btb] += adamW(llt,hhtw,btb,0,5);
		break;
		case 6:
			encoders[llte][hhte] += adamW(llt,hhte,0,0,6);
		break;
	}
	
}

function trainlinear(input,label) {
	
	costpertoken = maketensor(2,[layers+1,hiddensize],0,false);//hidden
	costpertoken[layers] = runexample(input,label);//output
	for (bb = layers; bb >= 0; bb--) {//layer
		for (aa = 0; aa < weights[bb].length; aa++) {//first neuron
			for (aa1 = 0; aa1 < weights[bb][aa].length; aa1++) {//second neuron
				let gfd = getfuncderiv(neuronstore[bb+1][aa1]);
				weights[bb][aa][aa1] += //sum of
					activate([neuronstore[bb][aa]])[0] * //in terms of zl- prev neuron is what influences zl
					gfd * //in terms of al- derivative of relu w/ respect to zl
					costpertoken[bb][aa1] *  //in terms of cost- desired change to cost
					learningrate;
				if (bb != 0) {
					biases[bb-1][aa] += 
						1 * //in terms of zl- bias does not influence zl
						gfd * //in terms of al- derivative of prev w/ respect to zl
						costpertoken[bb][aa1] *  //in terms of cost- desired change to cost down the line
						learningrate;
					costpertoken[bb-1][aa] += 
						weights[bb][aa][aa1] * //in terms of zl- weight is what influences zl
						gfd * //in terms of al- derivative of relu w/ respect to zl
						costpertoken[bb][aa1];  //in terms of cost- desired change to cost down the line
				}//next costs if not final layer
			}
		}
	}

}//train the model

function activate(ARR) {
	
	let ra = [];
	for (g = 0; g < ARR.length; g++) {
		if (type == "sigmoid") {
			ra[g] = sigmoid(ARR[g]);
		}
		else if (type == "RELU") {
			ra[g] = RELU(ARR[g]);
		}
		else if (type == "GELU") {
			ra[g] = GELU(ARR[g]);
		}
	}
	return ra;
	
}

function softmax(ARR) {
	
	let exsum = 0;
	let arrtoreturn = [];
	for (g = 0; g < ARR.length; g++) {
		exsum += pow(e,ARR[g]/smtemperature);
	}
	for (g = 0; g < ARR.length; g++) {
		arrtoreturn[g] = pow(e,ARR[g]/smtemperature)/exsum;
	}
	return arrtoreturn;
	
}

function Bsort(ARR,sourceARR,softmaxb,highlow,byprop,prop) {
	
	if (softmaxb == true) {
		ARR = softmax(ARR);
	}
	
	let sorted = [];
	if (byprop != true) {
		sorted = transpose([sourceARR,ARR]);
	}
	else {
		sorted = ARR;
	}
	
	let arrp = 1;
	if (byprop == true) {
		arrp = prop;
	}
	if (highlow == true) {
		sorted.sort(function(p, q) {
			return q[arrp] - p[arrp];
		});
	}
	else {
		sorted.sort(function(p, q) {
			return p[arrp] - q[arrp];
		});
	}

	return sorted;
	
}//[0] is decoded value, [1] is strength of that value

function normalize(ARR,scalar) {
	
	let arrneg = mult2d([CA(ARR)],[maketensor(1,[ARR.length],-1)])[0];
	let nv = max(max(ARR),max(arrneg))/scalar;//find maximum value
	return div2d([CA(ARR)],[maketensor(1,[ARR.length],nv)])[0];
	
}

function getrandin(ARR,low,top,checksum) {
	
	let counter = 0;
	for (g = low; g < top; g++) {
		if (ARR[g] === checksum) {
			counter++;
		}
	}//how many checksums are there
	let randranged = rr(1,top-counter-low+1);//get a random number
	for (g = low; g < top; g++) {
		if (ARR[g] !== checksum) {
			randranged--;
		}
		if (randranged == 0) {
			return g;
		}
	}//number in between low and randr
	
}

function tril(ARR,v) {
	
	for (g = 0; g < ARR.length; g++) {
		for (g1 = 0; g1 < ARR[0].length; g1++) {
			if (g1 > g) {
				ARR[g][g1] = v;
			}
		}
	}
	
	return ARR;
	
}//create triangle of (v) values in 2d array

function transpose(ARR) {
	
	return ARR[0].map((_, colIndex) => ARR.map(row => row[colIndex]));
	
}//switch columns with rows, preserve values

function matrixmult(ARR1,ARR2) {
	
	if (ARR1[0].length != ARR2.length) {
		print("ERROR:\ninfo:\nARR1:",ARR1,"ARR2:",ARR2)
		exit()
	}
	let returnarr = [];
	for (g = 0; g < ARR1.length; g++) {
		returnarr[g] = [];
		for (g1 = 0; g1 < ARR2[0].length; g1++) {
			returnarr[g][g1] = 0;
		}
	}//initialize returnarrs
	
	for (g = 0; g < returnarr.length; g++) {
		for (g1 = 0; g1 < returnarr[g].length; g1++) {
			for (g2 = 0; g2 < ARR1[0].length; g2++) {
				returnarr[g][g1] += ARR1[g][g2]*ARR2[g2][g1];
			}
		}
	}
	
	return returnarr;
	
}//can also be used with ([vec1],transpose([vec2])) for dot product

function add2d(ARR1,ARR2) {
	
	let ra = ARR1;
	for (g = 0; g < ARR1.length; g++) {
		for (g1 = 0; g1 < ARR1[0].length; g1++) {
			ra[g][g1] += ARR2[g][g1];
		}
	}
	return ra;
	
}//adds two same size arrays

function sub2d(ARR1,ARR2) {
	
	let ra = ARR1;
	for (g = 0; g < ARR1.length; g++) {
		for (g1 = 0; g1 < ARR1[0].length; g1++) {
			ra[g][g1] -= ARR2[g][g1];
		}
	}
	return ra;
	
}//adds two same size arrays

function mult2d(ARR1,ARR2) {

	let ra = ARR1;
	for (g = 0; g < ARR1.length; g++) {
		for (g1 = 0; g1 < ARR1[0].length; g1++) {
			ra[g][g1] *= ARR2[g][g1];
		}
	}
	return ra;
	
}//adds two same size arrays

function div2d(ARR1,ARR2) {
	
	let ra = ARR1;
	for (g = 0; g < ARR1.length; g++) {
		for (g1 = 0; g1 < ARR1[0].length; g1++) {
			ra[g][g1] /= ARR2[g][g1];
		}
	}
	return ra;
	
}//adds two same size arrays

function concatenate(ARR) {
	
	let ra = [];
	for (g = 0; g < ARR.length; g++) {
		for (g1 = 0; g1 < ARR[g].length; g1++) {
			ra[ra.length] = ARR[g][g1];
		}
	}
	return ra;
	
}//combines rows of 2d array into 1d array

function CA(ARR,obj) {
	
	return ARR.slice(0);
	
}//prevents fucking awful javascript auto-pointers (copy array)

function GELU(num) {
	
	return scale*num/(1+pow(e,-1.702*num));
	
}

function sigmoid(num) {
	
	return (1/(1+(pow(e,-1*scale*num))))
	
}

function RELU(num) {
	
	return max(0,scale*num);
	
}

function rr(low,top) {
	
	if (low == top) {
		print("ur stupid","go fix ur code -xoxo, rr (low==top error)")
		return top;
	}
	return round(random(low-0.5,top-0.5));
	
}

function getfuncderiv(input) {
		
	if (type == "sigmoid") {
		return (scale*pow(e,-1*scale*input))/pow(1+pow(e,-1*scale*input),2);
	}
	else if (type == "RELU") {
		return 1;
	}
	else if (type == "GELU") {
		let ndx = pow(e,1.702*input);
		return (ndx*scale*(1+ndx+1.702*input))/pow(1+ndx,2)
	}

}

function untoken(Q) {
	
	for (g = 0; g < tokens.length; g++) {
		if (tokens[g] == Q) {
			return g;
		}
	}
	if (Q == "") {
		return 8;
	}//newline catch
	return -1;
	
}

function positioners(x) {
	
	let sx = learningset/1.57079633;
	return sin(x/sx);
	
}

function maketensor(dim,shapeARR,fill,ifrand,randl,randh,ifroundrand,ascending) {
	
	let ra = []
	for (g = 0; g < shapeARR[0] && dim > 0; g++) {
		ra[g] = [];
		for (g1 = 0; g1 < shapeARR[1] && dim > 1; g1++) {
			ra[g][g1] = [];
			for (g2 = 0; g2 < shapeARR[2] && dim > 2; g2++) {
				ra[g][g1][g2] = [];
				for (g3 = 0; g3 < shapeARR[3] && dim > 3; g3++) {
					ra[g][g1][g2][g3] = [];
					for (g4 = 0; g4 < shapeARR[4] && dim > 4; g4++) {
						ra[g][g1][g2][g3][g4] = [];
						for (g5 = 0; g5 < shapeARR[5] && dim > 5; g5++) {
							ra[g][g1][g2][g3][g4][g5] = getfill(g5);
						}
						if (dim == 5) {
							ra[g][g1][g2][g3][g4] = getfill(g4);
						}
					}
					if (dim == 4) {
						ra[g][g1][g2][g3] = getfill(g3);
					}
				}
				if (dim == 3) {
					ra[g][g1][g2] = getfill(g2);
				}
			}
			if (dim == 2) {
				ra[g][g1] = getfill(g1);
			}
		}
		if (dim == 1) {
			ra[g] = getfill(g);
		}
	}//initializes arrays
	
	function getfill(index) {
		if (ifrand == true) {
			if (ifroundrand == true) {
				return rr(randl,randh);
			}
			else {
				return random(randl,randh);
			}
		}
		else if (ascending == true) {
			return index;
		}
		else if (typeof fill === 'function') {
			return fill();
		}
		else {
			return fill;
		}
	}
	
	return ra;
	
}//limit of 6 dimensions, randl = lower bound, randh = upper bound

function shapenet(shapeARR,specific,dim,sizing,fill,ifrand,randl,randh,ifroundrand,ascending) {
	
	let totalshape;
	if (specific == false) {
		totalshape = [shapeARR[0]];
		for (gsn = 1; gsn < sizing+1; gsn++) {
			totalshape[gsn] = shapeARR[1];
		}
		totalshape[sizing+1] = shapeARR[2];
	}
	else {
		totalshape = shapeARR;
	}
	let rasn = [];
	if (dim == 1) {
		totalshape = transpose([totalshape])
	}
	for (gsn = 0; gsn < totalshape.length; gsn++) {
		rasn[gsn] = maketensor(dim,[totalshape[gsn][0],totalshape[gsn][1]],fill,ifrand,randl,randh,ifroundrand,ascending);
	}
	return rasn;
	
}
