function runGPT(input) {

	//multihead self attention
	last = input
	neuronstore = maketensor(3,[layers],0);//init
	for (let ll = 0; ll < layers; ll++) {
		//multihead self attention
		let fval = maketensor(3,[heads,learningset,querykeydim],0);//desired changes arr
		for (hh = 0; hh < heads; hh++) {
			let normtensor = tril(maketensor(2,[learningset,learningset],0),-INF);//init- [query][key]
			for (b = 0; b < learningset; b++) {
				//dot query with all input based keys
				let qdot = matrixmult([CA(last[b])],CA(query[ll][hh]));//gets query vec with matrix
				for (c = 0; c < learningset; c++) {
					if (normtensor[b][c] == -INF) {
						break;
					}//mask rest of row
					let encoded = add2d([CA(last[c])],[maketensor(1,[encodesize],positioners(c+1))]);//encode position into word vec
					let kdot = matrixmult(encoded,CA(key[ll][hh]));//gets corresponding key vec with matrix
					normtensor[b][c] = matrixmult(qdot,transpose(kdot))[0][0]/sqrt(querykeydim);//find vec association and scale (dot product)
				}

				//softmax vector associations
				normtensor[b] = softmax(normtensor[b]);

				//sums desired changes scaled by vec association
				for (c = 0; c < learningset; c++) {
					let curval = matrixmult([CA(last[c])],CA(value[ll][hh]))[0];//value*key encode gives (down: [qkdim])
					fval[hh][b] = add2d([CA(fval[hh][b])],mult2d([CA(curval)],maketensor(2,[1,querykeydim],normtensor[b][c])))[0];
				}
			}
		}//perform attention

		//add proposed to last
		let learnout = transpose(fval);//[learningset,heads,querykeydim]
		for (b = 0; b < learningset; b++) {
			learnout[b] = concatenate(learnout[b]);//[heads*querykeydim]
		}
		last = add2d(CA(last),learnout);
		
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
	if (biasesarr != false && biasesarr !== undefined) {
		returnarr = opxd("add",returnarr,biasesarr);
	}

	return returnarr;

}//takes in 1d array and returns one transform with weights from layer + bias

function runlinear(input,allweights,allbiases,sorted) {

	let nsra = [CA(input)];
	let ra = CA(input);
	if (!Array.isArray(allbiases)) {
	    allbiases = tensor(false,[allweights.length]);
	}
	for (let a = 0; a < allweights.length; a++) {
		nsra[a+1] = linear(CA(ra),allweights[a],allbiases[a]);
		ra = activate(CA(nsra[a+1]));
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
