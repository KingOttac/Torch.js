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

function linear(ARR,weightsarr,biasesarr) {

	let returnarr = matrixmult([ARR],weightsarr)[0];
	if (addbias == true) {
		returnarr = add2d([returnarr],[biasesarr])[0];
	}

	return returnarr;

}//takes in 1d array and returns one transform with weights from layer + bias

function runlinear(input,qlayers,sorted,allweights,allbiases) {

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
