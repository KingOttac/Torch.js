function adamW(p0,p1,p2,p3,pv) {
		
	function dimen(assign,arr,pv,val) {
		if (assign == true) {
			if (dsties[pv] == 2) {
				arr[p0][p1] = val;
			}//encoder changes
			else if (dsties[pv] == 3) {
				arr[p0][p1][p2] = val;
			}//bias changes
			else if (dsties[pv] == 4) {
				arr[p0][p1][p2][p3] = val;
			}//key, query, valdown, valup, weights
		}
		else {
			if (dsties[pv] == 2) {
				return arr[p0][p1];
			}//encoder changes
			else if (dsties[pv] == 3) {
				return arr[p0][p1][p2];
			}//bias changes
			else if (dsties[pv] == 4) {
				return arr[p0][p1][p2][p3];
			}//key, query, valdown, valup, weights
		}
	}//different dimensional arrays- assign: set to or return

	//move in from array
	let mtin = dimen(false,mt[pv],pv);
	let vtin = dimen(false,vt[pv],pv);
	let tspin = dimen(false,tsp[pv],pv);

	//calculate vec adjust
	tspin++;
	let randomspot = rr(learningset,convertedlines.length);
	let rereturn = runexample(randomspot);
	let gtin = cc[randomspot-learningset]-rereturn;
	mtin = mtin + gtin;//get first vec change
	vtin = vtin + pow(gtin,2);//get second vec change

	//move back changed values
	dimen(true,mt[pv],pv,mtin)
	dimen(true,vt[pv],pv,vtin)
	dimen(true,tsp[pv],pv,tspin)
	cc[randomspot-learningset] = rereturn;

	//send a value to param
	let rturn = -alpha*mtin/sqrt(vtin);
	if (isNaN(rturn) == true) {
		return 0;
	}
	else {
		return rturn;
	}

}
