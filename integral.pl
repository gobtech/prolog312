% CPSC 312 Calculus and Algebra in Prolog
% Copyright D. Poole 2019. Released under GPL https://www.gnu.org/licenses/gpl-3.0.en.html
  
% An expression can include algebaic variables, which are Prolog constants

%eval(Exp, Env, V) is true if expression Exp evaluates to V given environment Env
% An environment is a list of val(Var,Val) indicating that variable Var has value Val
eval(X,Env,V) :-
    member(val(X,V),Env).
eval(N,_,N) :-
    number(N).
eval((A+B),Env,V) :-
    eval(A,Env,VA),
    eval(B,Env,VB),
    V is VA+VB.
eval((A*B),Env,V) :-
    eval(A,Env,VA),
    eval(B,Env,VB),
    V is VA*VB.
eval((A-B),Env,V) :-
    eval(A,Env,VA),
    eval(B,Env,VB),
    V is VA-VB.
eval(-A,Env,V) :-
    eval(A,Env,VA),
    V is -VA.
eval((A^B),Env,V) :-
    eval(A,Env,VA),
    eval(B,Env,VB),
    V is VA^VB.
eval(log(A),Env,V) :-
    eval(A,Env,VA),
    V is log(VA).
eval(exp(A),Env,V) :-
    eval(A,Env,VA),
    V is exp(VA).
eval(sigmoid(A),Env,V) :-
    eval(A,Env,VA),
    V is 1/(1+exp(-VA)).

% try:
% eval(aa*aa+b*11, [val(aa,3), val(b,7), val(dd,23)], V).

%  Differentiation

% deriv(E,X,DE) is true if DE is the derivative of E with respect to X
deriv(X,X,1).
deriv(C,X,0) :- atomic(C), dif(C,X).
deriv(A+B,X,DA+DB) :-
    deriv(A,X,DA),
    deriv(B,X,DB).
deriv(A-B,X,DA-DB) :-
    deriv(A,X,DA),
    deriv(B,X,DB).
deriv(A*B,X,A*DB+B*DA) :-
    deriv(A,X,DA),
    deriv(B,X,DB).
deriv(A/B,X,(B*DA-A*DB)/(B*B)) :-
    deriv(A,X,DA),
    deriv(B,X,DB).

antiderivative(X,Y) :- antid(Z,x,Y),simplify(Z,X).

deriv(-A,X,-DA) :-
    deriv(A,X,DA).
deriv(A^B,X,B*(A^(B-1))*DA) :-  % only works when B does not involve X
    deriv(A,X,DA).
deriv(sin(E),X,cos(E)*DE) :-
    deriv(E,X,DE).
deriv(cos(E),X,-sin(E)*DE) :-
    deriv(E,X,DE).
deriv(exp(E),X,exp(E)*DE) :-
    deriv(E,X,DE).
deriv(log(E),X,DE/E) :-
  deriv(E,X,DE).
% sigmoid(X) = 1/(1+exp(-X))
deriv(sigmoid(E),X,sigmoid(E)*(1-sigmoid(E))*DE) :-
    deriv(E,X,DE).

% Some Examples to try:
%?- deriv(x+3*x+6*x*y, x, D).
%?- deriv(7+3*x+6*x*y, x, D).
%?- deriv(x+3*x+6*x*y+ 11*x*x, x, D).
%?- deriv(1/(1+exp(-x)),x,D), simp(D,E).

% Multi-variate calculus:
%?- deriv(x+3*x+6*x*y+ 11*x*x, x, Dx), deriv(Dx,y,Dxy), simplify(Dxy,E).


%simplify(Exp, Exp2) is true if expression Exp2 is a simplifed form of Exp
simplify(X,X) :-
    atomic(X).
simplify((A+B),V) :-
    simplify(A, VA),
    simplify(B, VB),
    simp_vals(VA+VB, V).
simplify((A*B),V) :-
    simplify(A, VA),
    simplify(B, VB),
    simp_vals(VA*VB, V).
simplify((A/B),V) :-
    simplify(A, VA),
    simplify(B, VB),
    simp_vals(VA/VB, V).
simplify((A-B),V) :-
    simplify(A, VA),
    simplify(B, VB),
    simp_vals(VA-VB, V).
simplify(-E,R) :-
    simplify(E,S),
    simp_vals(-S,R).
simplify(sigmoid(E),sigmoid(S)) :-
    simplify(E,S).
simplify(log(E),log(S)) :-
    simplify(E,S).
simplify(exp(E),exp(S)) :-
    simplify(E,S).
simplify(A^B,S^B) :-
    simplify(A,S).x

%simp_vals(Exp, Exp2) is true if expression Exp simplifies to Exp2,
% where the arguments to Exp have already been simplified
%note last clause is a catch-all.
simp_vals(0+V,V).
simp_vals(V+0,V).
simp_vals(V-0,V).
simp_vals(0-V,- V).
simp_vals(-(0),0).
simp_vals(-(-X),X).
simp_vals(A+B,AB) :-
    number(A),number(B),
    AB is A+B.
simp_vals(A-B,AB) :-
    number(A),number(B),
    AB is A-B.
simp_vals(0*_,0).
simp_vals(_*0,0).
simp_vals(_*(-0),0).
simp_vals(_*(-(0)),0).
simp_vals(V*1,V).
simp_vals(1*V,V).
simp_vals(A*B,AB) :-
    number(A),number(B),
    AB is A*B.
simp_vals(0/_,0).
simp_vals(V/1,V).
simp_vals(A/B,AB) :-
    number(A),number(B),
    AB is A/B.
simp_vals(X,X).

% try:
%?- simplify(y*1+(0*x + x*0),E).
%?- simplify(y*(2*10+3),E).
%?- simplify(1+ (3*1+x*0)+ (6*x*0+y* (6*1+x*0))+ (11*x*1+x* (11*1+x*0)), E).

% Examples from learning (some that I used when building a learning system)
% deriv(-log(sigmoid(w3)*phgr+sigmoid(w4)*(1-phgr)),w3,D), simplify(D,S).
% deriv(-log(sigmoid(w3)*phgr+sigmoid(w4)*(1-phgr)),w4,D), simplify(D,S).
% deriv(-log(sigmoid(w4) + (sigmoid(w3)-sigmoid(w4))*sigmoid(w0+w1*pr+w2*nr)),w0,D),simplify(D,S).
% % deriv(-log(sigmoid(w4) + (sigmoid(w3)-sigmoid(w4))*sigmoid(w0+w1*pr+w2*nr)),w1,D),simplify(D,S).

% deriv(-log(1-(sigmoid(w3)*phgr+sigmoid(w4)*(1-phgr))),w3,D), simplify(D,S).
% deriv(-log(1-(sigmoid(w4) + (sigmoid(w3)-sigmoid(w4))*sigmoid(w0+w1*pr+w2*nr))),w1,D),simplify(D,S).

% deriv(-y*log(1-(1-p3)*(1-p1)^pr*(1-p2)^nr)-(1-y)*log((1-p3)*(1-p1)^pr*(1-p2)^nr),p3,D), simplify(D,S).
% deriv(-y*log(1-(1-p3)*(1-p1)^pr*(1-p2)^nr),p3,D), simplify(D,S).
% deriv(-(1-y)*log((1-p3)*(1-p1)^pr*(1-p2)^nr),p3,D), simplify(D,S).

% integ(E,X,IE) is true if IE is the indefinite integral of E with respect to X

antid(1,X,X).
antid(0,_,0).
antid(F,DX,X) :- deriv(X,DX,DE), simplify(DE,F).
antid(A+B,X,DA+DB) :-
    deriv(DA,X,A),
    deriv(DB,X,B).
antid(DA-DB,X,A-B) :-
    deriv(A,X,DA),
    deriv(B,X,DB).
antid(A*DB+B*DA,X,A*B) :-
    deriv(A,X,DA),
    deriv(B,X,DB).
antid((B*DA-A*DB)/(B*B),X,A/B) :-
    deriv(A,X,DA),
    deriv(B,X,DB).

antid(-DA,X,-A) :- deriv(A,X,DA).

antid(B*(A^(B-1))*DA,X,A^B) :-  % only works when B does not involve X
    deriv(A,X,DA).

antid(cos(E)*DE,X,sin(E)) :-
        deriv(E,X,DE).
antid(sin(E)*DE,X,-cos(E)) :-
        deriv(E,X,DE).
antid(exp(E)*DE,X,exp(E)) :-
        deriv(E,X,DE).
antid(DE/E,X,log(E)) :-
      deriv(E,X,DE).
    % sigmoid(X) = 1/(1+exp(-X))
antid(sigmoid(E)*(1-sigmoid(E))*DE,X,sigmoid(E)) :-
        deriv(E,X,DE).

%antisimplify(Exp, Exp2) is true if expression Exp2 is a simplifed form of Exp
antisimplify(X,1*X^1).
antisimplify((C*X),(C*X^1)) :- atomic(C).
antisimplify((X^C),(1*X^C)) :- atomic(C).

% integ(E,X,IE) is true if IE is the indefinite integral of E with respect to X

% axioms
integ(C,X,C*X) :- atomic(C), dif(C,X).
integ(0,_,0).
integ(X,X,(1/2)*X^2).
integ(C*X,X,C*I) :- atomic(C), integ(X,X,I).
integ(X*C,X,C*I) :- atomic(C), integ(X,X,I).
integ((A/B)*X,X,(A/B)*I) :- atomic(A),atomic(B),integ(X,X,I).
integ(X*(A/B),X,(A/B)*I) :- atomic(A),atomic(B),integ(X,X,I).


integ(A+B,X,IA + IB):- integ(A,X,IA),integ(B,X,IB).
integ(A-B,X,IA - IB):- integ(A,X,IA),integ(B,X,IB).
integ(U*V,X,U*I) :- atomic(U), dif(U,X), integ(V,X,I).
integ(-A,X,-IA) :- integ(A,X,IA).

integ(A^X,X,(A^(X))/(ln(X))) :- atomic(A), dif(A,X).
integ(X^A,X,(1/(A+1))*X^(A+1)) :- atomic(A), dif(A,-1).
integ(X^(A/B),X,(B/(A+B))*X^((A+B)/B)) :- atomic(A), atomic(B), dif(A,B).
integ(A/X,X,A*ln(X)) :- atomic(A), dif(A,X).
integ(((A*X+B)^(-1) ) * X , X , (A*X-B*ln(A*X+B))/(a^2) ).
integ(((A*X+B)^N ) * X ,X,(( A*X + B) ^ ( N + 1))/( A * ( N + 1 ) )).
integ(X^A,X,(1/(A+1))*X^(A+1)) :- atomic(A), dif(A,-1).


% functions 
integ(exp(X),X,exp(X)).
integ(exp(B*X),X,exp(B*X)/B) :- atomic(B).
integ(ln(X),X,X*ln(X)-X).
integ(ln(B*X),X,X*ln(B*X)-X) :- atomic(B).
integ(cos(X),X,sin(X)).
integ(sin(X),X,-cos(X)).
integ(cos(A*X+B),X,sin(A*X+B)/A) :- atomic(A), atomic(B).
integ(sin(A*X+B),X,-cos(A*X+B)/A) :- atomic(A), atomic(B).
integ(sec(X)^2,X,tan(X)).
integ(sec(B*X)^2,X,tan(B*x)/B) :- atomic(B).
integ(sec(X),X ,ln(sec(X) + tan(X))).
integ(cosec(X), X, ln( cosec(X) - cot(X) ) ).
integ( cot(X) , X , ln(sin(X)) ).
integ(cosec(X)^2 , X , -cot(X) ).
integ( sec(X) * tan(X) , X , sec(X) ).
integ( cosec(X) * cot(X) , X , -cosec(X) ).
integ( 1/(A*X+B) , X , (1/A) * ln(A*X+B) ).
integ( exp(A*X+B) , X , (1/A) * exp(A*X+B) ).
integ( 1/sqrt(1-X^2) , X , arcsin(X) ).
integ( 1/sqrt(1+X^2) , X , arctan(X) ).
integ( 1/(X*sqrt(X^2-1)) , X , arcsec(X) ).
integ( 1/(X^2-A^2) , X , (1/2*A) * ln( (X-A) / (X+A) ) ).
integ( 1/(A^2-X^2) , X , (1/2*A) * ln( (A+X) / (A-X) ) ).
integ( 1/(X^2+A^2) , X , (1/A)* arctan(X/A) ).
integ( 1/sqrt(X^2-A^2) , X , ln( X+sqrt(X^2-A^2) ) ).
integ( 1/sqrt(A^2-X^2) , X , arcsin(X/A) ).
integ( 1/sqrt(A^2+X^2) , X , ln(X+sqrt(X^2+A^2)) ).

integ(U*DV,X,U*V- IVDU) :- deriv(V,X,DV), deriv(U,X,DU), integ(V*DU,X,IVDU).
