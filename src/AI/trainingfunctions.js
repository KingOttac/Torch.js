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
