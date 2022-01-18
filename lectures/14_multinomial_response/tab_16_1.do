 use keane.dta
 mlogit status educ exper expersq black if year==87, base(0)
 margins, dydx(educ) predict(outcome(0)) predict(outcome(1))  predict(outcome(2))
 
