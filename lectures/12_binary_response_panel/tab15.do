clear all
use "lfp.dta"
bysort id: egen kidsbar=mean(kids)
bysort id: egen lhincbar=mean(lhinc)
* Replication of resulst in Table 15.3 in Wooldridge 2010
* COl 1
xtreg lfp kids lhinc per2-per5, fe cluster(id)
* COl 2
probit lfp kids lhinc educ black age agesq per2-per5, cluster(id)
* COl 3
probit lfp kids lhinc kidsbar lhincbar educ black age agesq per2-per5, cluster(id)
* COl 4 
* RE results in Wooldridge does not appear to have converged 
* (has lower likelihood and too low sigma_u - perhaps Stata fixed a bug since 2010)
xtprobit lfp kids lhinc kidsbar lhincbar educ black age agesq per2-per5, re
* COl 5
xtlogit lfp kids lhinc per2-per5, fe
