params <-
list(presentation = TRUE)

#' ---
#' title: Session 4
#' subtitle: Multi-test, multi-population models
#' date: "2023-06-08"
#' author:
#'   - Matt Denwood, Giles Innocent
#' theme: metropolis
#' aspectratio: 169
#' colortheme: seahorse
#' header-includes: 
#'   - \input{preamble}
#' params:
#'   presentation: TRUE
#' output:
#'   html_document: default
#'   beamer_presentation:
#'       pandoc_args: ["-t", "beamer"]
#'       slide_level: 2
#' ---
#' 
## ----setup, include=FALSE-----------------------------------------------------
source("setup.R")

#' 
#' ## Why stop at two tests?
#' 
#' In *traditional* diagnostic test evaluation, one test is assumed to be a gold standard from which all other tests are evaluated
#' 
#'   - So it makes no difference if you assess one test at a time or do multiple tests at the same time
#' 
#' . . .
#' 
#' Using a latent class model each new test adds new information - so we should analyse all available test results in the same model
#' 
#' ## Simulating data: simple example
#' 
#' Simulating data using an arbitrary number of independent tests is quite straightforward:
#' 
## -----------------------------------------------------------------------------
# Parameter values to simulate:
N <- 200
sensitivity <- c(0.8, 0.9, 0.95)
specificity <- c(0.95, 0.99, 0.95)

Populations <- 2
prevalence <- c(0.25,0.5)

data <- tibble(Population = sample(seq_len(Populations), N, replace=TRUE)) %>%
  mutate(Status = rbinom(N, 1, prevalence[Population])) %>%
  mutate(Test1 = rbinom(N, 1, sensitivity[1]*Status + (1-specificity[1])*(1-Status))) %>%
  mutate(Test2 = rbinom(N, 1, sensitivity[2]*Status + (1-specificity[2])*(1-Status))) %>%
  mutate(Test3 = rbinom(N, 1, sensitivity[3]*Status + (1-specificity[3])*(1-Status))) %>%
  select(-Status)


#' 
#' 
#' ## Model specification
#' 
#' Like for two tests, except it is now a 2x2x2 table
#'  
#' . . .
#' 
## ----include=FALSE, eval=FALSE------------------------------------------------
## template_huiwalter(data)
## cleanup <- c(cleanup, "huiwalter_model.txt")

#' 
## ----eval=FALSE---------------------------------------------------------------
## 
## Tally[1:8,p] ~ dmulti(prob[1:8,p], TotalTests[p])
## 
## # Probability of observing Test1- Test2- Test3-
## prob[1,p] <-  prev[p] * ((1-se[1])*(1-se[2])*(1-se[3]) +
##               (1-prev[p]) * (sp[1]*sp[2]*sp[3])
## 
## # Probability of observing Test1+ Test2- Test3-
## prob[2,p] <-  prev[p] * (se[1]*(1-se[2])*(1-se[3])) +
##               (1-prev[p]) * ((1-sp[1])*sp[2]*sp[3])
## 
## ## snip ##
## 
## # Probability of observing Test1+ Test2+ Test3+
## prob[3,p] <-  prev[p] * (se[1]*se[2]*se[3]) +
##               (1-prev[p]) * ((1-sp[1])*(1-sp[2])*(1-sp[3]))

#' 
#' . . .
#' 
#' - We need to take **extreme** care with these equations, and the multinomial tabulation!!!
#'  
#' 
#' ## Are the tests conditionally independent?
#' 
#' 
#' - Example:  we have one blood, one milk, and one faecal test
#' 
#'   * But the blood and milk test are basically the same test
#'   * Therefore they are more likely to give the same result
#'   
#' . . .
#' 
#' - Example:  we test people for COVID using an antigen test on a nasal swab, a PCR test on a throat swab, and the same antigen test on the same throat swab
#' 
#'   * The virus may be present in the throat, nose, neither, or both
#'   * But we use the same antigen test twice
#'     * Might it cross-react with the same non-target virus?
#' 
#' . . .
#' 
#' - In both situations we have pairwise correlation between some of the tests
#' 
#' 
#' ## Directed Acyclic Graphs
#' 
#' - It may help you to visualise the relationships as a DAG:
#' 
## ----echo=FALSE, fig.height=4-------------------------------------------------
covid_dag <- dagify(
  infected ~ prevalence,
  virus_throat ~ infected,
  virus_nose ~ infected,
  throat_pcr ~ virus_throat,
  throat_antigen ~ virus_throat,
  nose_antigen ~ virus_nose,
#  throat_antigen ~ cross_reaction,
#  nose_antigen ~ cross_reaction,
  latent = c("infected", "virus_throat", "virus_nose"),
  exposure = "prevalence",
  outcome = c("throat_pcr", "throat_antigen", "nose_antigen"),
  labels = c("infected"="infected", "prevalence"="prevalence",
             "virus_throat"="virus_throat", "virus_nose"="virus_nose",
             "throat_pcr"="throat_pcr", "throat_antigen"="throat_antigen",
             "nose_antigen"="nose_antigen", "cross_reaction"="cross_reaction")
)
ggdag(covid_dag, text=FALSE, use_labels="label") + theme_dag_blank()

#' 
#' 
#' ## Dealing with correlation: Covid example
#' 
#' It helps to consider the data simulation as a (simplified) biological process (where my parameters are not representative of real life!):
#' 
## -----------------------------------------------------------------------------
# The probability of infection with COVID in two populations:
prevalence <- c(0.01,0.05)
# The probability of shedding COVID in the nose conditional on infection:
nose_shedding <- 0.8
# The probability of shedding COVID in the throat conditional on infection:
throat_shedding <- 0.8
# The probability of detecting virus with the antigen test:
antigen_detection <- 0.75
# The probability of detecting virus with the PCR test:
pcr_detection <- 0.999
# The probability of random cross-reaction with the antigen test:
antigen_crossreact <- 0.05
# The probability of random cross-reaction with the PCR test:
pcr_crossreact <- 0.01

#' 
#' . . .
#' 
#' Note:  cross-reactions are assumed to be independent here!
#' 
#' ---
#' 
#' Simulating latent states:
#' 
## -----------------------------------------------------------------------------
N <- 20000
Populations <- length(prevalence)

covid_data <- tibble(Population = sample(seq_len(Populations), N, replace=TRUE)) %>%
  ## True infection status:
  mutate(Status = rbinom(N, 1, prevalence[Population])) %>%
  ## Nose shedding status:
  mutate(Nose = Status * rbinom(N, 1, nose_shedding)) %>%
  ## Throat shedding status:
  mutate(Throat = Status * rbinom(N, 1, throat_shedding))

#' 
#' - - -
#' 
#' Simulating test results:
#' 
## -----------------------------------------------------------------------------
covid_data <- covid_data %>%
  ## The nose swab antigen test may be false or true positive:
  mutate(NoseAG = case_when(
    Nose == 1 ~ rbinom(N, 1, antigen_detection),
    Nose == 0 ~ rbinom(N, 1, antigen_crossreact)
  )) %>%
  ## The throat swab antigen test may be false or true positive:
  mutate(ThroatAG = case_when(
    Throat == 1 ~ rbinom(N, 1, antigen_detection),
    Throat == 0 ~ rbinom(N, 1, antigen_crossreact)
  )) %>%
  ## The PCR test may be false or true positive:
  mutate(ThroatPCR = case_when(
    Throat == 1 ~ rbinom(N, 1, pcr_detection),
    Throat == 0 ~ rbinom(N, 1, pcr_crossreact)
  ))

#' 
#' ---
#' 
#' The overall sensitivity of the tests can be calculated as follows:
#' 
## -----------------------------------------------------------------------------
covid_sensitivity <- c(
  # Nose antigen:
  nose_shedding*antigen_detection + (1-nose_shedding)*antigen_crossreact,
  # Throat antigen:
  throat_shedding*antigen_detection + (1-throat_shedding)*antigen_crossreact,
  # Throat PCR:
  throat_shedding*pcr_detection + (1-throat_shedding)*pcr_crossreact
)
covid_sensitivity

#' 
#' - - -
#' 
#' The overall specificity of the tests is more straightforward:
#' 
## -----------------------------------------------------------------------------
covid_specificity <- c(
  # Nose antigen:
  1 - antigen_crossreact,
  # Throat antigen:
  1 - antigen_crossreact,
  # Throat PCR:
  1 - pcr_crossreact
)
covid_specificity

#' 
#' . . .
#' 
#' However:  this assumes that cross-reactions are independent!
#' 
#' 
#' ## Model specification
#' 
## ---- eval=FALSE--------------------------------------------------------------
## prob[1,p] <-  prev[p] * ((1-se[1])*(1-se[2])*(1-se[3])
##                          +covse12 +covse13 +covse23) +
##               (1-prev[p]) * (sp[1]*sp[2]*sp[3]
##                              +covsp12 +covsp13 +covsp23)
## 
## prob[2,p] <- prev[p] * (se[1]*(1-se[2])*(1-se[3])
## 	                       -covse12 -covse13 +covse23) +
## 	           (1-prev[p]) * ((1-sp[1])*sp[2]*sp[3]
## 	                          -covsp12 -covsp13 +covsp23)
## 
## ## snip ##
## 		
## # Covariance in sensitivity between tests 1 and 2:
## covse12 ~ dunif( (se[1]-1)*(1-se[2]) ,
## 	                 min(se[1],se[2]) - se[1]*se[2] )
## # Covariance in specificity between tests 1 and 2:
## covsp12 ~ dunif( (sp[1]-1)*(1-sp[2]) ,
## 	                 min(sp[1],sp[2]) - sp[1]*sp[2] )

#' 
#' . . .
#' 
#' It is quite easy to get the terms slightly wrong!
#' 
#' ## Template Hui-Walter
#' 
#' The model code and data format for an arbitrary number of populations (and tests) can be determined automatically using the template_huiwalter function from the runjags package:
#' 
## ----results='hide'-----------------------------------------------------------
template_huiwalter(
  covid_data %>% select(Population, NoseAG, ThroatAG, ThroatPCR), 
  outfile = 'covidmodel.txt')

#' 
#' This generates self-contained model/data/initial values etc
#' 
#' ---
#' 
## ----echo=FALSE, comment=''---------------------------------------------------
cleanup <- c(cleanup, 'covidmodel.txt')
cat(readLines('covidmodel.txt')[3:150], sep='\n')

#' 
#' ---
#' 
## ----echo=FALSE, comment=''---------------------------------------------------
cat(readLines('covidmodel.txt')[-(1:150)], sep='\n')

#' 
#' ---
#' 
#' And can be run directly from R:
#' 
## ---- results='hide'----------------------------------------------------------
results <- run.jags('covidmodel.txt')
results

#' 
## ----echo=FALSE---------------------------------------------------------------
res <- summary(results)[c(1:8,seq(9,19,by=2)),c(1:3,9,11)]
res[] <- round(res, 3)
knitr::kable(res)

#' 
#' ## Template Hui-Walter
#' 
#' - Modifying priors must still be done directly in the model file
#'   * Same for adding .RNG.seed and the deviance monitor
#' 
#' - The model needs to be re-generated if the data changes
#'   * But remember that your modified priors will be reset
#' 
#' - There must be a single column for the population (as a factor), and all of the other columns (either factor, logical or numeric) are interpreted as being test results
#' 
#' - - -
#' 
#' - Covariance terms are also calculated as proportion of possible correlation e.g.:
#' 
## ----echo=FALSE, comment=''---------------------------------------------------
cat(readLines('covidmodel.txt')[(87:91)], sep='\n')

#' 
#' . . .
#' 
#' - But covariance terms are all deactivated by default!
#' 
#' 
#' ## Activating covariance terms
#' 
#' Find the lines for the covariances that we want to activate (i.e. the two Throat tests):
#' 
## ---- echo=FALSE, comment=''--------------------------------------------------
indexes <- c(111:113, 116:119)
cleanup <- c(cleanup, "covidmodel.txt")
ml <- readLines('covidmodel.txt')
cat(gsub('\t','',ml[indexes]), sep='\n')

#' 
#' ---
#' 
#' And edit so it looks like:
#' 
## ---- echo=FALSE, comment=''--------------------------------------------------
ml[indexes] <- c('	# Covariance in sensitivity between ThroatAG and ThroatPCR tests:', '	covse23 ~ dunif( (se[2]-1)*(1-se[3]) , min(se[2],se[3]) - se[2]*se[3] )  ## if the sensitivity of these tests may be correlated', '	 # covse23 <- 0  ## if the sensitivity of these tests can be assumed to be independent','','	# Covariance in specificity between ThroatAG and ThroatPCR tests:', '	covsp23 ~ dunif( (sp[2]-1)*(1-sp[3]) , min(sp[2],sp[3]) - sp[2]*sp[3] )  ## if the specificity of these tests may be correlated', '	 # covsp23 <- 0  ## if the specificity of these tests can be assumed to be independent')
cat(ml, file='covidmodel.txt', sep='\n')
ml <- readLines('covidmodel.txt')
cat(gsub('\t','',ml[indexes]), sep='\n')

#' [i.e. swap the comments around]
#' 
#' ---
#' 
#' You will also need to uncomment out the relevant initial values for BOTH chains (on lines 132-137 and 128-133):
#' 
## ---- echo=FALSE, comment=''--------------------------------------------------
ml <- readLines('covidmodel.txt')
cat(gsub('\t','',ml[143:148]), sep='\n')

#' 
#' So that they look like:
#' 
## ---- echo=FALSE, comment=''--------------------------------------------------
ml[c(134,137)] <- c('"covse23" <- 0', '"covsp23" <- 0')
ml[c(145,148)] <- c('"covse23" <- 0', '"covsp23" <- 0')
cat(ml, file='covidmodel.txt', sep='\n')
ml <- readLines('covidmodel.txt')
cat(gsub('\t','',ml[143:148]), sep='\n')

#' 
#' ---
#' 
## -----------------------------------------------------------------------------
results <- run.jags('covidmodel.txt', sample=50000)
results

#' 
#' 
#' ## Practical considerations
#' 
#' - Correlation terms add complexity to the model in terms of:
#'   * Opportunity to make a coding mistake
#'   * Reduced identifiability
#' 
#' . . .
#' 
#' - The template_huiwalter function helps us with coding mistakes
#' 
#' - Only careful consideration of covariance terms can help us with identifiability
#' 
#' 
#' 
#' # How to interpret the latent class
#' 
#' ## A hierarchy of latent states
#' 
#' 
## ----echo=FALSE, fig.height=4-------------------------------------------------
ab_dag <- dagify(
  infected ~ prevalence,
  antibodies ~ infected,
  abtarget ~ antibodies,
  abtarget2 ~ antibodies,
  test1 ~ abtarget,
  test2 ~ abtarget,
  test3 ~ abtarget2,
  latent = c("infected", "antibodies", "abtarget","abtarget2"),
  exposure = "prevalence",
  outcome = c("test1", "test2", "test3"),
  labels = c("infected"="Infected", "prevalence"="Prevalence",
             "antibodies"="Producing Antibodies", "abtarget"="Presence of Target 1","abtarget2"="Presence of Target 2",
             "test1"="ELISA A", "test2"="ELISA B", "test3"="ELISA C")
)
ggdag(ab_dag, text=FALSE, use_labels="label") + theme_dag_blank()

#' 
#' 
#' ## What is sensitivity and specificity?
#' 
#' - The probability of test status conditional on true disease status?
#' 
#' - The probability of test status conditional on the latent state?
#' 
#' . . .
#' 
#' So is the latent state the same as the true disease state?
#' 
#' . . .
#' 
#' Important quote:
#' 
#' "Latent class models involve pulling **something** out of a hat, and deciding to call it a rabbit"
#' 
#'   - Nils Toft
#' 
#' 
#' 
#' 
#' ## When should we correct for correlation?
#' 
#' - For each of the following DAG:
#' 
#'   * Consider what is the latent class
#' 
#'   * Consider which correlation terms we should include and why
#' 
#' 
#' - - -
#' 
## ----echo=FALSE, fig.height=4-------------------------------------------------
eg_dag <- dagify(
  infected ~ prevalence,
#  antibodies ~ infected,
#  abtarget ~ antibodies,
  test1 ~ infected,
  test2 ~ infected,
  test3 ~ infected,
  latent = c("infected"),
  exposure = "prevalence",
  outcome = c("test1", "test2", "test3"),
  labels = c("infected"="Infected", "prevalence"="Prevalence",
             "antigen"="Pathogen Detectable", "antibodies"="Producing Antibodies", "abtarget"="Presence of Target",
             "test1"="Test A", "test2"="Test B", "test3"="Test C")
)
ggdag(eg_dag, text=FALSE, use_labels="label") + theme_dag_blank()

#' 
#' . . .
#' 
#' - No correlation to model!
#' 
#' - - -
#' 
## ----echo=FALSE, fig.height=4-------------------------------------------------
eg_dag <- dagify(
  infected ~ prevalence,
  antibodies ~ infected,
#  abtarget ~ antibodies,
  test1 ~ antibodies,
  test2 ~ antibodies,
  test3 ~ antibodies,
  latent = c("infected","antibodies"),
  exposure = "prevalence",
  outcome = c("test1", "test2", "test3"),
  labels = c("infected"="Infected", "prevalence"="Prevalence",
             "antigen"="Pathogen Detectable", "antibodies"="Producing Antibodies", "abtarget"="Presence of Target",
             "test1"="Test A", "test2"="Test B", "test3"="Test C")
)
ggdag(eg_dag, text=FALSE, use_labels="label") + theme_dag_blank()

#' 
#' . . .
#' 
#' - No correlation to model ... but "infected" is not the latent class
#' 
#' - - -
#' 
## ----echo=FALSE, fig.height=4-------------------------------------------------
eg_dag <- dagify(
  infected ~ prevalence,
  antibodies ~ infected,
  abtarget ~ antibodies,
  test1 ~ abtarget,
  test2 ~ abtarget,
  test3 ~ abtarget,
  latent = c("infected","antibodies","abtarget"),
  exposure = "prevalence",
  outcome = c("test1", "test2", "test3"),
  labels = c("infected"="Infected", "prevalence"="Prevalence",
             "antigen"="Pathogen Detectable", "antibodies"="Producing Antibodies", "abtarget"="Presence of Target",
             "test1"="Test A", "test2"="Test B", "test3"="Test C")
)
ggdag(eg_dag, text=FALSE, use_labels="label") + theme_dag_blank()

#' 
#' . . .
#' 
#' - Same as above!
#' 
#' 
#' - - -
#' 
## ----echo=FALSE, fig.height=4-------------------------------------------------
eg_dag <- dagify(
  infected ~ prevalence,
  antigen ~ infected,
  antibodies ~ infected,
  abtarget ~ antibodies,
  test1 ~ antigen,
  test2 ~ abtarget,
  test3 ~ abtarget,
  latent = c("infected","antibodies","abtarget"),
  exposure = "prevalence",
  outcome = c("test1", "test2", "test3"),
  labels = c("infected"="Infected", "prevalence"="Prevalence",
             "antigen"="Pathogen Detectable", "antibodies"="Producing Antibodies", "abtarget"="Presence of Target",
             "test1"="Test A", "test2"="Test B", "test3"="Test C")
)
ggdag(eg_dag, text=FALSE, use_labels="label") + theme_dag_blank()

#' 
#' . . .
#' 
#' - Tests B and C are correlated
#' 
#' - - -
#' 
## ----echo=FALSE, fig.height=4-------------------------------------------------
eg_dag <- dagify(
  infected ~ prevalence,
  antigen ~ infected,
  antibodies ~ infected,
  abtarget1 ~ antibodies,
  abtarget2 ~ antibodies,
  test1 ~ abtarget1,
  test2 ~ abtarget2,
  test3 ~ abtarget2,
  latent = c("infected","antibodies","abtarget1","abtarget2"),
  exposure = "prevalence",
  outcome = c("test1", "test2", "test3"),
  labels = c("infected"="Infected", "prevalence"="Prevalence",
             "antigen"="Pathogen Detectable", "antibodies"="Producing Antibodies", "abtarget1"="Presence of Target 1", "abtarget2"="Presence of Target 2",
             "test1"="Test A", "test2"="Test B", "test3"="Test C")
)
ggdag(eg_dag, text=FALSE, use_labels="label") + theme_dag_blank()

#' 
#' . . .
#' 
#' - All tests are correlated with respect to infected BUT infected is not the latent class
#' 
#' - Tests B and C are correlated with respect to antibodies - but maybe not substantially?
#' 
#' - - -
#' 
## ----echo=FALSE, fig.height=4-------------------------------------------------
eg_dag <- dagify(
  infected ~ prevalence,
  antigen ~ infected,
  antibodies ~ infected,
  test1 ~ antigen,
  test2 ~ antibodies,
  latent = c("infected","antibodies"),
  exposure = "prevalence",
  outcome = c("test1", "test2"),
  labels = c("infected"="Infected", "prevalence"="Prevalence",
             "antigen"="Pathogen Detectable", "antibodies"="Producing Antibodies", "abtarget"="Presence of Target",
             "test1"="Test A", "test2"="Test B", "test3"="Test C")
)
ggdag(eg_dag, text=FALSE, use_labels="label") + theme_dag_blank()

#' 
#' . . .
#' 
#' - No correlation to model
#' 
#' - - -
#' 
## ----echo=FALSE, fig.height=4-------------------------------------------------
eg_dag <- dagify(
  infected ~ prevalence,
#  antigen ~ infected,
  antibodies ~ infected,
  test1 ~ antibodies,
  test2 ~ antibodies,
  latent = c("infected","antibodies"),
  exposure = "prevalence",
  outcome = c("test1", "test2"),
  labels = c("infected"="Infected", "prevalence"="Prevalence",
             "antigen"="Pathogen Detectable", "antibodies"="Producing Antibodies", "abtarget"="Presence of Target",
             "test1"="Test A", "test2"="Test B", "test3"="Test C")
)
ggdag(eg_dag, text=FALSE, use_labels="label") + theme_dag_blank()

#' 
#' . . .
#' 
#' - No correlation to model - but "infected" is not the latent class
#' 
#' 
#' ## Publication of your results
#' 
#' STARD-BLCM:  A helpful structure to ensure that papers contain all necessary information
#'   
#'   - You should follow this and refer to it in your articles!
#' 
#' . . .
#' 
#' If you use the software, please cite JAGS:
#' 
#'   - Plummer, M. (2003). JAGS : A Program for Analysis of Bayesian Graphical Models Using Gibbs Sampling JAGS : Just Another Gibbs Sampler. Proceedings of the 3rd International Workshop on Distributed Statistical Computing (DSC 2003), March 20–22,Vienna, Austria. ISSN 1609-395X. https://doi.org/10.1.1.13.3406
#' 
#' ---
#' 
#' And R:
#' 
## -----------------------------------------------------------------------------
citation()

#' 
#' ---
#' 
#' And runjags:
#' 
## -----------------------------------------------------------------------------
citation("runjags")

#' 
#' 
#' 
#' # Practical session 4
#' 
#' ## Points to consider {.fragile}
#' 
#' 1. How does including a third test impact the inference for the first two tests?
#' 
#' 1. What happens if we include correlation between tests?
#' 
#' 
#' `r exercise_start()`
#' 
#' ## Exercise 1 {.fragile}
#' 
#' Use the template_huiwalter function to look at the simple 2-test 5-population example from session 3.  Use this data simulation code:
#' 
## -----------------------------------------------------------------------------
# Set a random seed so that the data are reproducible:
set.seed(2022-09-13)

sensitivity <- c(0.9, 0.6)
specificity <- c(0.95, 0.9)
N <- 1000

# Change the number of populations here:
Populations <- 5
# Change the variation in prevalence here:
(prevalence <- runif(Populations, min=0.1, max=0.9))

data <- tibble(Population = sample(seq_len(Populations), N, replace=TRUE)) %>%
  mutate(Status = rbinom(N, 1, prevalence[Population])) %>%
  mutate(Test1 = rbinom(N, 1, sensitivity[1]*Status + (1-specificity[1])*(1-Status))) %>%
  mutate(Test2 = rbinom(N, 1, sensitivity[2]*Status + (1-specificity[2])*(1-Status))) %>%
  select(-Status)

(twoXtwoXpop <- with(data, table(Test1, Test2, Population)))
(Tally <- matrix(twoXtwoXpop, ncol=Populations))
(TotalTests <- apply(Tally, 2, sum))

template_huiwalter(data, outfile="template_2test.txt")

#' 
#' Look at the model code and familiarise yourself with how the model is set out (there are some small differences, but the overall code is equivalent).  Make sure you can modify the priors and add a deviance monitor.  Run the model.
#' 
#' Now activate the correlation terms between tests 1 and 2.  Is anything different about the results?
#' 
#' ### Solution 1 {.fragile}
#' 
#' There is no particular solution to the first part of this exercise, but please ask if you have any questions about the model code that template_huiwalter generates.  Remember that re-running the template_huiwalter function will over-write your existing model including any changes you made, so be careful!
#' 
#' We can run the model as follows:
#' 
## -----------------------------------------------------------------------------
results_nocov <- run.jags("template_2test.txt")
results_nocov

## ----include=FALSE------------------------------------------------------------
cleanup <- c(cleanup, "template_2test.txt")
cleanup <- c(cleanup, "template_2test_cov.txt")

#' 
#' A shortcut for activating the covariance terms is to re-run template_huiwalter as follows:
#' 
## -----------------------------------------------------------------------------
template_huiwalter(data, outfile="template_2test_cov.txt", covariance=TRUE)
results_cov <- run.jags("template_2test_cov.txt")
results_cov

#' 
#' Activating the covariance terms with 2 tests has made the model less identifiable, and has therefore decreased the effective sample size and increased the width of the 95% CI for all of the parameters to the point that the model is no longer very useful.  This is not something that we recommend you do in practice, even if the two tests are known to be correlated!  We will revisit this issue tomorrow.
#' 
#' ## Exercise 2 {.fragile}
#' 
#' Simulate some Covid data based on the R code given above (under "Dealing with correlation: Covid example") and analyse the data using the default priors.  Interpret the results and compare them to the known values:
#' 
## -----------------------------------------------------------------------------
covid_sensitivity
covid_specificity

#' 
#' Now exclude the NoseAG test from the dataset, re-generate the model code (without covariance terms), run the model, and interpret the results.  How have the posteriors for the throat tests been affected by excluding the nose swab test?
#' 
#' ### Solution 2 {.fragile}
#' 
#' Here are the results for all 3 tests:
#' 
## -----------------------------------------------------------------------------
results_3t <- run.jags('covidmodel.txt')
results_3t

#' 
#' I have omitted the trace plots here but make sure to always check them!  You can see that the true values for overall sensitivity and specificity are within their respective 95% CI:
#' 
## -----------------------------------------------------------------------------
covid_sensitivity
covid_specificity

#' 
#' You should also notice that the model has detected a positive covariance between tests 2 and 3 (the throat swab tests), although the 95% CI does include zero.  Estimating covariance terms is extremely difficult for the model to do accurately.
#' 
#' Excluding the nasal swab test gives us these results:
#' 
## -----------------------------------------------------------------------------
template_huiwalter(covid_data %>% select(Population, ThroatAG, ThroatPCR), outfile = 'covidmodel_2t.txt')

results_2t <- run.jags('covidmodel_2t.txt')
results_2t

#' 
## ----include=FALSE------------------------------------------------------------
cleanup <- c(cleanup, 'covidmodel_2t.txt')

#' 
#' The posterior estimates for sensitivity have been affected for both tests, and neither now identify the true simulation parameter:
#' 
## -----------------------------------------------------------------------------
test2 <- combine.mcmc(results_2t, vars="se", return.samples = 10000)
test3 <- combine.mcmc(results_3t, vars="se[2:3]", return.samples = 10000)

bind_rows(
  tibble(Model = "TwoTest", ThroatAG = test2[,1], ThroatPCR = test2[,2]),
  tibble(Model = "ThreeTest", ThroatAG = test3[,1], ThroatPCR = test3[,2])
) %>%
  pivot_longer(c(ThroatAG, ThroatPCR), names_to = "Test", values_to = "Estimate") %>%
  ggplot() +
  aes(x = Estimate, col = Model) +
  geom_density() +
  facet_wrap( ~ Test)

#' 
#' However, they do look more similar to the probabilities of detection conditional on shedding:
#' 
## -----------------------------------------------------------------------------
antigen_detection
pcr_detection

#' 
#' Have a think about why this might be the case.  We will spend a lot of time discussing this topic tomorrow!
#' 
#' 
#' ## Optional exercise A {.fragile}
#' 
#' Re-fit a model to this data using all three possible covse and covsp parameters between all 3 tests
#' 
#' What do you notice about the results?
#' 
#' ### Solution A {.fragile}
#' 
#' You can either manually change all 3 covse/covsp from before, or regenerate the model using the covariance=TRUE option:
#' 
## ---- results='hide'----------------------------------------------------------
template_huiwalter(covid_data %>% select(Population, NoseAG, ThroatAG, ThroatPCR), outfile='covidmodel_allcov.txt', covariance=TRUE)
results_allcov <- run.jags('covidmodel_allcov.txt')

#' 
## ----echo=FALSE---------------------------------------------------------------
cleanup <- c(cleanup, 'covidmodel_allcov.txt')

#' 
## -----------------------------------------------------------------------------
results_allcov

#' 
#' The effective sample size is much lower, because the model is less identifiable.  But otherwise the model does a reasonable job of estimating the parameters due to the large sample size, albeit with wider 95% CI then with only covariance between throat swab tests included  This might not be the case with a smaller sample size!
#' 
#' `r exercise_end()`
#' 
#' 
#' ## Summary {.fragile}
#' 
#' - Including multiple tests is technically easy
#'   - But philosophically more difficult!!!
#' 
#' - Complexity of adding correlation terms increases non-linearly with more tests
#'   - Probably best to stick to correlations with biological justification?
#'   
#' - Adding/removing test results may change the posterior for
#'   - Other test Se / Sp
#'   - Prevalence
#' 
#' 
## ----include=FALSE------------------------------------------------------------
unlink(cleanup)

