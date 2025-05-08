function runGPT(input) {

	//multihead self attention
	last = input
	neuronstore = maketensor(3,[layers],0);//init
	for (let ll = 0; ll < layers; ll++) {
		//multihead self attention
		let fval = maketensor(3,[heads,learningset,encodesize],0);
		for (let hh = 0; hh < heads; hh++) {
			let normtensor = tril(maketensor(2,[learningset,learningset],0),-INF);//init- [query][key]
			for (let b = 0; b < learningset; b++) {
				let qdot = matrixmult([CA(last[b])],CA(query[ll][hh]));//gets query vec with matrix
				for (let c = 0; c < learningset; c++) {
					if (normtensor[b][c] == -INF) {
						break;
					}//mask rest of row
					let encoded = add2d([CA(last[c])],[maketensor(1,[encodesize],positioners(c+1))]);//encode position into word vec
					let kdot = matrixmult(encoded,CA(key[ll][hh]));//gets corresponding key vec with matrix
					normtensor[b][c] = matrixmult(qdot,transpose(kdot))[0][0]/sqrt(querykeydim);//find vec association and scale (basically dot product)
				}
			}//dot all queries with all input based keys
			for (let b = 0; b < learningset; b++) {
				normtensor[b] = softmax(normtensor[b]);//softmax vector associations
			}//softmax normtensor rows
			for (let a = 0; a < learningset; a++) {
				//get vector shift with scaled value vec
				let curval = matrixmult([CA(last[a])],CA(valuedown[ll][hh]));//valuedown*key encode gives (down: [qkdim])
				curval = matrixmult(curval,CA(valueup[ll][hh]))[0];//valueup*curval gives (back up to [encode])
				for (let b = 0; b < learningset; b++) {
					fval[hh][a] = add2d([CA(fval[hh][a])],mult2d([CA(curval)],maketensor(2,[1,encodesize],normtensor[a][b])))[0];
				}
			}//update desired changes to last
		}//perform attention
		for (let hh = 0; hh < heads; hh++) {
			last = add2d(last,fval[hh]);
		}//edit last
		
		//feed forward network
		for (let hh = 0; hh < learningset; hh++) {
			last[hh] = normalize(last[hh],wi);
			let linout = runlinear(CA(last[hh]),ffnlayers+1,false,weights[ll],biases[ll])//gets neuron arrangement[1] and new output flow values[0]
			last[hh] = normalize(add2d([last[hh]],[linout[0]])[0],wi);//add and normalize
		}
	}//multilayer gpt oh yeah
	
	return matrixmult([last[last.length-1]],transpose(encoders))[0];//[0] is just to lower dimension
	
}//takes in previous tokens as numbers

function linear(ARR,weightsarr,biasesarr) {

	let returnarr = matrixmult([ARR],weightsarr)[0];
	if (biasesarr !== undefined) {
		returnarr = add2d([returnarr],[biasesarr])[0];
	}

	return returnarr;

}//takes in 1d array and returns one transform with weights from layer + bias

function runlinear(input,allweights,allbiases,sorted) {

	let nsra = [];
	nsra[0] = input;
	let ra = input;
	for (let a = 0; a < allweights.length; a++) {
		nsra[a+1] = linear(CA(ra),allweights[a],allbiases[a]);
		ra = activate(nsra[a+1]);
	}
	if (sorted == true) {
		return [Bsort(CA(ra),outputarr),nsra];
	}
	else {
		return [CA(ra),nsra];
	}

}//takes in parameters for layers (corr to weights), if sorted, input 
//returns an array of [output , each layer unactivated arr]

function runGen(input,net,sorted,actin,outputarr) {
	
	if (actin == true) {
		input = activate(input);
	}//activate input
	for (let gg = 0; gg < inputsize; gg++) {
		net.neurons[0][gg] = input[gg];
	}//insert input
	for (let gg = 1; gg < net.neurons.length; gg++) {//layers
		for (let gg1 = 0; gg1 < net.neurons[gg].length; gg1++) {//neurons in layer
			net.neurons[gg][gg1] = 0;
		}
	}//zero out rest of network
	for (let gg = 0; gg < net.neurons.length-1; gg++) {//layers
		net.neurons[gg+1] = activate(linear(net.neurons[gg],net.weights[gg],net.biases[gg]));
	}//apply weights and biases
	
	//returning
	let returnarr = CA(net.neurons[net.neurons.length-1]);
	if (sorted == true) {
		return Bsort(returnarr,outputarr,false);
	}
	else {
		return returnarr;
	}
	
}
