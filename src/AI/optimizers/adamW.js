function adamW(pv,p0,p1,p2,p3,dst) {

	//move in from array
	let mtin = dimen(false,mt[pv],dst);
	let vtin = dimen(false,vt[pv],dst);
	let tspin = dimen(false,tsp[pv],dst);

	//calculate vec adjust
	tspin++;
	let randomspot = rr(learningset,convertedlines.length);
	let rereturn = runexample(randomspot);
	let gtin = cc[randomspot-learningset]-rereturn;
	mtin = mtin + gtin;//get first vec change
	vtin = vtin + pow(gtin,2);//get second vec change

	//move back changed values
	dimen(true,mt[pv],dst,mtin)
	dimen(true,vt[pv],dst,vtin)
	dimen(true,tsp[pv],dst,tspin)
	cc[randomspot-learningset] = rereturn;

	//send a value to param
	let rturn = -alpha*mtin/sqrt(vtin);
	if (isNaN(rturn) == true) {
		return 0;
	}
	else {
		return rturn;
	}

}//network consts
