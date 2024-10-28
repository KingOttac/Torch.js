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
let params = [];
let tsp = 0;

//linear
let neuronstore = [];
let weights = [];
let biases = [];
let costarr = [];

//generative
let scores = [];//generation scores
let currentbest = 0;

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

function loadshitlinear() {

	weights = shapenet([inputsize,hiddensize,outputsize],false,2,layers,0,true,-wi,wi);
	biases = shapenet([hiddensize,hiddensize,outputsize],false,1,layers-1,0,true,-wi,wi);

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

function runGen(input,sorted,iid,outputarr,allnetworks) {
	
	for (gg = 0; gg < inputsize; gg++) {
		if (activatein == true)	 {
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

function dimen(assign,arr,parr,val) {
	let p0 = parr[0];
	let p1 = parr[1];
	let p2 = parr[2];
	let p3 = parr[3];
	let p4 = parr[4];
	let p5 = parr[5];
	let dst = parr.length;
	if (assign == true) {
		if (dst == 1) {
			arr[p0] = val;
		}//1D
		else if (dst == 2) {
			arr[p0][p1] = val;
		}//2D
		else if (dst == 3) {
			arr[p0][p1][p2] = val;
		}//3D
		else if (dst == 4) {
			arr[p0][p1][p2][p3] = val;
		}//4D
		else if (dst == 5) {
			arr[p0][p1][p2][p3][p4] = val;
		}//5D
		else if (dst == 6) {
			arr[p0][p1][p2][p3][p4][p5] = val;
		}//6D
	}
	else {
		if (dst == 1) {
			return arr[p0];
		}//1D
		else if (dst == 2) {
			return arr[p0][p1];
		}//2D
		else if (dst == 3) {
			return arr[p0][p1][p2];
		}//3D
		else if (dst == 4) {
			return arr[p0][p1][p2][p3];
		}//4D
		else if (dst == 5) {
			return arr[p0][p1][p2][p3][p4];
		}//5D
		else if (dst == 6) {
			return arr[p0][p1][p2][p3][p4][p5];
		}//6D
	}
}//different dimensional arrays- assign: bool- t:assign or f:return

function adamW(parr,gtin) {

	//move in from array
	let mtin = dimen(false,mt,parr);
	let vtin = dimen(false,vt,parr);
	let tspin = tsp;
	
	//calculate vec adjust
	mtin = b1*mtin + (1-b1)*gtin;//get first vec change
	vtin = b2*vtin + (1-b2)*pow(gtin,2);//get second vec change
	let mtv = mtin/(1-pow(b1,tspin));//first vec bias correct
	let vtv = vtin/(1-pow(b2,tspin));//second vec bias correct
	
	//move back changed values
	dimen(true,mt,parr,mtin);
	dimen(true,vt,parr,vtin);

	//send a value to param
	return -alpha*mtv/(sqrt(vtv)+epsilon);

}//network consts

function trainGPT() {
	
	tsp++;
	for (ee = 0; ee < epoch; ee++) {
		for (bb = 0; bb < batch; bb++) {
			let randin = rr(0,convertedlines.length)
			let gcost = runexample(randin);
			for (llt = 0; llt < layers; llt++) {
				for (hht = 0; hht < heads; hht++) {
					for (bt = 0; bt < encodesize; bt++) {
						for (ct = 0; ct < querykeydim; ct++) {
							key[llt][hht][bt][ct] += adamW([0,llt,hht,bt,ct],gcost);
							query[llt][hht][bt][ct] += adamW([1,llt,hht,bt,ct],gcost);
							valuedown[llt][hht][bt][ct] += adamW([2,llt,hht,bt,ct],gcost);
							valueup[llt][hht][ct][bt] += adamW([3,llt,hht,ct,bt],gcost);
						}
					}
				}//k,q,vup,vdown
				for (hht = 0; hht < ffnlayers; hht++) {
					for (bt = 0; bt < weights[llt][hht].length; bt++) {
						for (ct = 0; ct < weights[llt][hht][bt].length; ct++) {
							weights[llt][hht][bt][ct] += adamW([4,llt,hht,bt,ct],gcost);
						}
					}//weights
					for (bt = 0; bt < biases[llt][hht].length; bt++) {
						biases[llt][hht][bt] += adamW([5,llt,hht,bt],gcost);
					}//biases
				}//weights,biases
			}
			for (llt = 0; llt < tokens.length; llt++) {
				for (hht = 0; hht < encodesize; hht++) {
					encoders[llt][hht] += adamW([6,llt,hht],gcost);
				}
			}
		}
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
		let fval = maketensor(3,[heads,learningset,encodesize],0);//desired changes arr
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
			}//update desired changes arr
		}//perform attention
		for (hh = 0; hh < heads; hh++) {
			last = add2d(CA(last),CA(fval[hh]));
		}//add proposed to last
		
		//feed forward network
		for (hh = 0; hh < learningset; hh++) {
			last[hh] = normalize(last[hh],wi);
			let linout = runlinear(CA(last[hh]),ffnlayers+1,false,weights[ll],biases[ll])//gets neuron arrangement[1] and new output flow values[0]
			last[hh] = normalize(add2d([last[hh]],[linout[0]])[0],wi);//add and normalize
		}
	}//multilayer gpt oh yeah
	
	return matrixmult([CA(last[last.length-1])],transpose(encoders))[0];
	
}//takes in previous tokens as numbers

function trainlinear() {
	
	costpertoken = maketensor(2,[layers+1,hiddensize],0);//hidden
	costpertoken[layers] = maketensor(1,[outputsize],0);//output
	runexample();
	for (bb = layers; bb >= 0; bb--) {//layer
		for (aa = 0; aa < weights[bb].length; aa++) {//first neuron
			for (aa1 = 0; aa1 < weights[bb][aa].length; aa1++) {//second neuron
				weights[bb][aa][aa1] += //sum of
					activate([neuronstore[bb][aa]])[0] * //in terms of zl- prev neuron is what influences zl
					getfuncderiv(neuronstore[bb+1][aa1]) * //in terms of al- derivative of relu w/ respect to zl
					costpertoken[bb][aa1] *  //in terms of cost- desired change to cost
					learningrate;
				if (bb != 0) {
					biases[bb-1][aa] += 
						1 * //in terms of zl- bias does not influence zl
						getfuncderiv(neuronstore[bb+1][aa1]) * //in terms of al- derivative of prev w/ respect to zl
						costpertoken[bb][aa1] *  //in terms of cost- desired change to cost down the line
						learningrate;
					costpertoken[bb-1][aa] += 
						weights[bb][aa][aa1] * //in terms of zl- weight is what influences zl
						getfuncderiv(neuronstore[bb+1][aa1]) * //in terms of al- derivative of relu w/ respect to zl
						costpertoken[bb][aa1];  //in terms of cost- desired change to cost down the line
				}//next costs if not final layer
			}
		}
	}//calculate new weights+biases

}//train the model

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
	
}//one dimentional note

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
							ra[g][g1][g2][g3][g4][g5] = getfill([g,g1,g2,g3,g4,g5]);
						}
						if (dim == 5) {
							ra[g][g1][g2][g3][g4] = getfill([g,g1,g2,g3,g4]);
						}
					}
					if (dim == 4) {
						ra[g][g1][g2][g3] = getfill([g,g1,g2,g3]);
					}
				}
				if (dim == 3) {
					ra[g][g1][g2] = getfill([g,g1,g2]);
				}
			}
			if (dim == 2) {
				ra[g][g1] = getfill([g,g1]);
			}
		}
		if (dim == 1) {
			ra[g] = getfill([g]);
		}
	}//initializes arrays
	
	function getfill(parr) {
		if (ifrand == true) {
			if (ifroundrand == true) {
				return rr(randl,randh+1);
			}
			else {
				return random(randl,randh);
			}
		}
		else if (ascending == true) {
			return parr[parr.length-1];
		}
		else if (typeof fill === 'function') {
			return fill(parr);
		}
		else {
			return fill;
		}
	}
	
	return ra;
	
}//limit of 6 dimensions, randl = lower bound, randh = upper bound

function shapenet(shapeARR,specific,dim,sizing,fill,ifrand,randl,randh,ifroundrand,ascending) {
	
	let totalshape = [];
	if (specific == false) {
		totalshape[0] = [shapeARR[0],shapeARR[1]];
		for (gsn = 1; gsn < sizing; gsn++) {
			totalshape[gsn] = [shapeARR[1],shapeARR[1]];
		}
		totalshape[sizing] = [shapeARR[1],shapeARR[2]];
	}
	else {
		totalshape = shapeARR;
	}
	let rasn = [];
	if (dim == 1) {
		for (gsn = 0; gsn < totalshape.length; gsn++) {
			totalshape[gsn] = [totalshape[gsn][1]];
		}
	}//change for bias arrays
	for (gsn = 0; gsn < totalshape.length; gsn++) {
		rasn[gsn] = maketensor(dim,[totalshape[gsn][0],totalshape[gsn][1]],fill,ifrand,randl,randh,ifroundrand,ascending);
	}
	return rasn;
	
}

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

function normalize(ARR,scalar) {
	
	let arrneg = mult2d([CA(ARR)],[maketensor(1,[ARR.length],-1)])[0];
	let nv = max(max(ARR),max(arrneg))/scalar;//find maximum value
	return div2d([CA(ARR)],[maketensor(1,[ARR.length],nv)])[0];
	
}

function rr(low,top) {
	
	if (low == top) {
		print("ur stupid","go fix ur code -xoxo, rr (low==top error)")
		return top;
	}
	return round(random(low-0.5,top-0.5));
	
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

function unshift(arr,val) {
	
	let returnarr = [val];
	for (g = 0; g < arr.length; g++) {
		returnarr[g+1] = arr[g];
	}
	return returnarr;
	
}
