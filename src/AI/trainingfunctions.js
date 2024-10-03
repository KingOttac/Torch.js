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
