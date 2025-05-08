function trainGPT(disp) {
	
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
	let hhtd;
	if (params[7] !== undefined) {
		hhtd = rr(0,outputsize);
	}

	//choose param and run optimizer
	let pxrand = rr(0,params.length);
	let indexparrs = [
		[pxrand,llt,hht,bt,ct],
		[pxrand,llt,hht,bt,ct],
		[pxrand,llt,hht,bt,ct],
		[pxrand,llt,hht,ct,bt],
		[pxrand,llt,hhtw,btw,ctw],
		[pxrand,llt,hhtw,btb],
		[pxrand,llte,hhte],
		[pxrand,hhte,hhtd]
	];
	adamW(CA(indexparrs[pxrand]));
	runexample(getinput(),disp==0);//for display
	
}

function trainlinear(tps) {
	
	let costs = runexample(getinput(),tps == 0);//output
	backprop(weights,biases,neuronstore,costs);

}//train the model

function backprop(allweights,allbiases,nstore,costpertoken) {
	
	for (let bb = allweights.length-1; bb >= 0; bb--) {//layer
		let newcosts = maketensor(1,[allweights[bb].length],0);
		for (let aa = 0; aa < allweights[bb].length; aa++) {//first neuron
			for (let aa1 = 0; aa1 < allweights[bb][aa].length; aa1++) {//second neuron
				let gfd = getfuncderiv(nstore[bb+1][aa1]);
				allweights[bb][aa][aa1] += 
					activate([nstore[bb][aa]])[0] * //in terms of zl- prev neuron is what influences zl
					gfd * //in terms of al- derivative of relu w/ respect to zl
					costpertoken[aa1] *  //in terms of cost- desired change to cost
					learningrate;
				if (allbiases !== false) {
					allbiases[bb][aa1] += 
						gfd * //in terms of al- derivative of prev w/ respect to zl
						costpertoken[aa1] *  //in terms of cost- desired change to cost down the line
						learningrate;
				}
				newcosts[aa] += 
					allweights[bb][aa][aa1] * //in terms of zl- weight is what influences zl
					gfd * //in terms of al- derivative of relu w/ respect to zl
					costpertoken[aa1];  //in terms of cost- desired change to cost down the line
			}
		}
		costpertoken = newcosts;
	}
	return costpertoken;
	
}

function trainGen() {
	
	//bests
	let tsbests = Bsort(CA(scores),maketensor(1,[iterations],0,false,0,0,false,true),false,true);
	let newbest = false;
	if (tsbests[0][1] > totalbest) {
		totalbest = tsbests[0][1];
		newbest = true;
	}
	currentbest = tsbests[0][0];
	
	//evolve networks (copies to second to take only top half)
	let final = [];
	for (let a = 0; a < neuronstore.length; a++) {
		if (newbest) {
			final[a] = CA(neuronstore[currentbest]);
		}
		else {
			final[a] = CA(neuronstore[a]);
		}
		
		//get randoms
		let randlay = rr(0,final[a].neurons.length-1);
		let cwei = final[a].weights[randlay];
		let cbia = final[a].biases[randlay];
		let randc = random(0,1);
		if (randc < perc[0]) {
			randc = "newweight";
		}
		else if (randc >= 1-perc[1]-perc[2] && randc < 1-perc[2]) {
			randc = "newbias";
		}
		else {
			randc = "newneur";
		}
		
		//assign changes
		switch (randc) {
			case "newweight":
				final[a].weights[randlay][rr(0,cwei.length)][rr(0,cwei[0].length)] = random(-wi,wi);
			break;
			case "newbias":
				final[a].biases[randlay][rr(0,cbia.length)] = random(-wi,wi);
			break;
			case "newneur":
				let cnl = final[a].neurons.length-2;
				if (final[a].neurons[cnl].length == hiddensize || cnl == 0) {
					final[a].neurons[cnl+2] = tensor(0,[outputsize]);
					final[a].neurons[cnl+1] = tensor(0,[outputsize]);
					final[a].weights[cnl+1] = tensor(1,[outputsize,outputsize]);
					final[a].biases[cnl+1] = tensor(0,[outputsize]);
				}//new layer
				else {
					final[a].neurons[cnl][final[a].neurons[cnl].length] = 0;//neuron
					final[a].biases[cnl-1][final[a].biases[cnl-1].length] = 0;//bias
					final[a].weights[cnl][final[a].weights[cnl].length] = tensor(0,[outputsize]);//last weights
					for (let b = 0; b < final[a].weights[cnl-1].length; b++) {
						final[a].weights[cnl-1][b][final[a].weights[cnl-1][b].length] = 0;
					}//prev layer
				}//add to existing layer
			break;
		}
	}
	
	//apply and reset
	neuronstore = final;
	scores = maketensor(1,[iterations],0);
	
}
