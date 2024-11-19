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

	//choose param set and init
	let pxrand = rr(0,7);
	let indexparrs = [
		[pxrand,llt,hht,bt,ct],
		[pxrand,llt,hht,bt,ct],
		[pxrand,llt,hht,bt,ct],
		[pxrand,llt,hht,ct,bt],
		[pxrand,llt,hhtw,btw,ctw],
		[pxrand,llt,hhtw,btb],
		[pxrand,llt,hhte]
	];
	let tsp = 0;
	let mt = 0;
	let vt = 0;
	
	for (ee = 0; ee < epoch; ee++) {
		for (bch = 0; bch < batch; bch++) {
			
			//declare
			tsp++;
			let parr = indexparrs[pxrand];
			let randin = rr(learningset,learningset+sampleset);//choose train batch
			let gtin;
			
			//get gtin
			let gcost = runexample(randin,false);
			dimen(true,params,parr,dimen(false,params,parr)+alpha);//small change
			if (runexample(randin,false) > gcost) {
				gtin = -alpha;
			}//move backward
			else {
				gtin = alpha;
			}//move forward
			dimen(true,params,parr,dimen(false,params,parr)-alpha);//reset change

			//add back adam values
			let returnval = adamW(parr,gtin,mt,vt,tsp);
			mt = returnval[0];
			vt = returnval[1];
			dimen(true,params,parr,dimen(false,params,parr)+gtin)
			
		}
	}//perform grad descent
	let randin = rr(learningset,learningset+sampleset);
	runexample(randin,true);//for display
	
	/*
 	all parameters at once (broken)
	for (llt = 0; llt < layers; llt++) {
		for (hht = 0; hht < heads; hht++) {
			for (bt = 0; bt < encodesize; bt++) {
				for (ct = 0; ct < querykeydim; ct++) {
					key[llt][hht][bt][ct] += adamW([0,llt,hht,bt,ct]);
					query[llt][hht][bt][ct] += adamW([1,llt,hht,bt,ct]);
					valuedown[llt][hht][bt][ct] += adamW([2,llt,hht,bt,ct]);
					valueup[llt][hht][ct][bt] += adamW([3,llt,hht,ct,bt]);
				}
			}
		}//k,q,vup,vdown
		for (hht = 0; hht < ffnlayers; hht++) {
			for (bt = 0; bt < weights[llt][hht].length; bt++) {
				for (ct = 0; ct < weights[llt][hht][bt].length; ct++) {
					weights[llt][hht][bt][ct] += adamW([4,llt,hht,bt,ct]);
				}
			}//weights
			for (bt = 0; bt < biases[llt][hht].length; bt++) {
				biases[llt][hht][bt] += adamW([5,llt,hht,bt]);
			}//biases
		}//weights,biases
	}
	for (llt = 0; llt < tokens.length; llt++) {
		for (hht = 0; hht < encodesize; hht++) {
			encoders[llt][hht] += adamW([6,llt,hht]);
		}
	}
	*/
	
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
