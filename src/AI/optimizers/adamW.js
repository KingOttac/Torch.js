function adamW(parr) {
	
	let tsp = 0;
	let mt = 0;
	let vt = 0;
	for (let ee = 0; ee < epoch; ee++) {
		for (let bch = 0; bch < batch; bch++) {
			
			//declare
			tsp++;
			let randin = getinput();//load training data
			let gtin;
			
			//get gtin
			let gcost = runexample(randin,false);
			dimen(true,params,parr,dimen(false,params,parr)+alpha);//small change
			gtin = -(runexample(randin,false)-gcost)/alpha;//approximate deriv
			dimen(true,params,parr,dimen(false,params,parr)-alpha);//reset change

			//calculate vec adjust
			mt = b1*mt + (1-b1)*gtin;//get first vec change
			vt = b2*vt + (1-b2)*pow(gtin,2);//get second vec change
			let mtv = mt/(1-pow(b1,tsp));//first vec bias correct
			let vtv = vt/(1-pow(b2,tsp));//second vec bias correct

			//add back adam value
			let finalval = alpha*mtv/(sqrt(vtv)+epsilon);
			dimen(true,params,parr,dimen(false,params,parr)+finalval);
			
		}
	}//perform grad descent
	
}//network consts
