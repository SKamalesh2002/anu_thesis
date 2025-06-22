import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import mannwhitneyu, ttest_ind, ks_2samp, chi2_contingency
from scipy.stats import kruskal, ranksums, wilcoxon
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Anu's Medical Data Analysis Dashboard",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("🏥 Anu's Medical Data Analysis Dashboard")
st.markdown("### Multiple Statistical Tests Analysis for Clinical Outcomes")

# CSV data embedded as variable
csv_data = """NAME,PID NO,AGE,SEX,COMPLAINTS,K/C/O,SBP,DBP,SBP/DBP,SPO2%,RR,CBG,TEMPERATURE,HR,UREA,CREATININE,CRP,INITIAL LACTATE,REPEAT LACTATE,LACTATE CLEARANCE,CLINICAL OUTCOMES
GOPINATH,250302308432,45,MALE,BREATHLESSNESS,CKD SHTN,90 mm Hg,60 mm Hg,90/60 mm Hg,95%,40 min⁻¹,265 mg/dL,96.4 °F,148 BPM,24 mg/dL,1.0 mg/dL,39 mg/L,4.4 mmol/L,0.9 mmol/L,17.00%,ALIVE
SUBRAMANI,240517207052,84,MALE,BREATHLESSNESS,CKD SHTN,120 mm Hg,80 mm Hg,120/80 mm Hg,85%,33 min⁻¹,190 mg/dL,98.3 °F,133 BPM,96 mg/dL,4.0 mg/dL,359 mg/L,6.1 mmol/L,9.8 mmol/L,6.13%,DEAD
ARUMUGAM ,250123294432,59,MALE,AFI,CKD SHTN,110 mm Hg,70 mm Hg,110/70 mm Hg,95%,22 min⁻¹,197 mg/dL,100.8 °F,116 BPM,82 mg/dL,3.0 mg/dL,194 mg/L,1.2 mmol/L,0.7 mmol/L,41.67%,ALIVE
RAJENDRAN,241029265807,69,MALE,ALTERED SENSORIUM,CKD SHTN,100 mm Hg,60 mm Hg,100/60 mm Hg,83%,36 min⁻¹,145 mg/dL,97.4 °F,98 BPM,178 mg/dL,2.0 mg/dL,141 mg/L,4.2 mmol/L,5.8 mmol/L,8.20%,DEAD
DURAI,240704221812,55,MALE,BREATHLESSNESS,CKD SHTN,130 mm Hg,80 mm Hg,130/80 mm Hg,99%,21 min⁻¹,201 mg/dL,97.3 °F,84 BPM,147 mg/dL,3.0 mg/dL,48 mg/L,5.2 mmol/L,2.5 mmol/L,21.00%,ALIVE
CHITRA,230823121929,31,FEMALE,VOMITING,CKD T2DM SHTN,190 mm Hg,90 mm Hg,190/90 mm Hg,100%,25 min⁻¹,198 mg/dL,98.8 °F,79 BPM,58 mg/dL,5.0 mg/dL,71 mg/L,0.9 mmol/L,0.5 mmol/L,44.44%,ALIVE
VIJAYALAKSHMI,250224306358,57,FEMALE,BREATHLESSNESS,CKD T2DM SHTN,130 mm Hg,80 mm Hg,130/80 mm Hg,68%,42 min⁻¹,333 mg/dL,98.1 °F,92 BPM,47 mg/dL,3.0 mg/dL,57 mg/L,16.0 mmol/L,23. mmol/L,2.10%,DEAD
MADHURAI MUTHU,250306310374,90,MALE,BREATHLESSNESS, CKD T2DM SHTN,100 mm Hg,70 mm Hg,100/70 mm Hg,84%,43 min⁻¹,78 mg/dL,98.4 °F,144 BPM,20 mg/dL,1.0 mg/dL,45 mg/L,2.2 mmol/L,1.6 mmol/L,27.10%,ALIVE
VETRIVEL,250313313273,22,MALE,PEDAL EDEMA,CKD T2DM SHTN,140 mm Hg,110 mm Hg,140/110 mm Hg,100%,22 min⁻¹,102 mg/dL,98.1 °F,128 BPM,78 mg/dL,7.0 mg/dL,10 mg/L,2.0 mmol/L,0.8 mmol/L,19.00%,ALIVE
JAYACHANDRAN,250211300996,77,MALE,GIDDINESS,CKD T2DM SHTN,140 mm Hg,70 mm Hg,140/70 mm Hg,99%,24 min⁻¹,321 mg/dL,98.1 °F,84 BPM,116 mg/dL,5.0 mg/dL,11 mg/L,2.0 mmol/L,1.2 mmol/L,40.00%,ALIVE
BASKARAN,250127295238,63,MALE,BREATHLESSNESS AFI, CKD T2DM SHTN,140 mm Hg,80 mm Hg,140/80 mm Hg,97%,25 min⁻¹,196 mg/dL,104.5 °F,112 BPM,36 mg/dL,1.0 mg/dL,186 mg/L,3.2 mmol/L,2.4 mmol/L,25.00%,ALIVE
NATARAJAN,230628103362,56,MALE,BREATHLESSNESS,CKD APE SHTN,120 mm Hg,80 mm Hg,120/80 mm Hg,96%,27 min⁻¹,158 mg/dL,99.2 °F,94 BPM,142 mg/dL,8.0 mg/dL,213 mg/L,4.3 mmol/L,5.4 mmol/L,2.50%,DEAD
RAJAN DHARMA,2210042043,63,MALE,PEDAL EDEMA,CKD T2DM SHTN,110 mm Hg,70 mm Hg,110/70 mm Hg,97%,22 min⁻¹,125 mg/dL,92.5 °F,87 BPM,76 mg/dL,3.0 mg/dL,70 mg/L,1.4 mmol/L,1.2 mmol/L,14.29%,ALIVE
SANJEEVI,250324317166,53,MALE,DECREASED URINE OUTPUT,CKD SHTN,110 mm Hg,70 mm Hg,110/70 mm Hg,98%,22 min⁻¹,133 mg/dL,98.1 °F,95 BPM,158 mg/dL,7.0 mg/dL,69 mg/L,1.2 mmol/L,0.7mmol/l,42.00%,ALIVE
GANESAN,2210126346,64,MALE,BREATHLESSNESS,CKD T2DM CAD,190 mm Hg,100 mm Hg,190/100 mm Hg,99%,36 min⁻¹,224 mg/dL,98.1 °F,106 BPM,78 mg/dL,4.0 mg/dL,5 mg/L,0.9 mmol/L,0.7 mmol/L,22.00%,ALIVE
RUKIA KHATUN,250322316868,51,FEMALE,ABDOMINAL PAIN,CKD T2DM,110 mm Hg,70 mm Hg,110/70 mm Hg,98%,20 min⁻¹,340 mg/dL,98.7 °F,106 BPM,110 mg/dL,8.0 mg/dL,397 mg/L,1.0 mmol/L,0.9 mmol/L,10.00%,ALIVE
ETHIRAJULU,240531211189,63,MALE,LEFT LL WEAKNESS,CKD T2DM,110 mm Hg,70 mm Hg,110/70 mm Hg,98%,16 min⁻¹,242 mg/dL,97.6 °F,80 BPM,86 mg/dL,3.0 mg/dL,4,2.1 mmol/L,1.9 mmol/L,10.00%,ALIVE
NASIRA BEGUM,250314313963,47,FEMALE,BREATHLESSNESS,CKD LEPTOSPIROSIS,160 mm Hg,110 mm Hg,160/110 mm Hg,94%,23 min⁻¹,134 mg/dL,98.6 °F,118 BPM,151 mg/dL,5.0 mg/dL,231 mg/L,2.8 mmol/L,1.9 mmol/L,32.00%,ALIVE
MOHAMMAD ISMAIL,241231287687,69,MALE,AFI DISORIENTED,CKD T2DM CAD,150 mm Hg,80 mm Hg,150/80 mm Hg,94%,20 min⁻¹,94 mg/dL,97.9 °F,64 BPM,129 mg/dL,5.0 mg/dL,192 mg/L,1.7 mmol/L,1.2 mmol/L,29.00%,ALIVE
ELUMALAI,241230287659,74,MALE,CHEST PAIN,CKD T2DM SHTN,130 mm Hg,80 mm Hg,130/80 mm Hg,100%,22 min⁻¹,124 mg/dL,97.4 °F,74 BPM,111 mg/dL,4.0 mg/dL,10 mg/L,1.6 mmol/L,1.4 mmol/L,13.00%,ALIVE
DHANA BAKKIYAM,241220284534,79,FEMALE,GIDDINESS,CKD T2DM SHTN,200 mm Hg,100 mm Hg,200/100 mm Hg,97%,24 min⁻¹,81 mg/dL,96.6 °F,68 BPM,103 mg/dL,6.0 mg/dL,6 mg/L,1.4 mmol/L,0.9 mmol/L,36.00%,ALIVE
RAMCHANDRA RAUT,241215282950,64,MALE,SCROTAL SWELLING,CKD T2DM SHTN,140 mm Hg,90 mm Hg,140/90 mm Hg,99%,22 min⁻¹,HIGH,98.1 °F,80 BPM,97 mg/dL,3.0 mg/dL,431 mg/L,4.6 mmol/L,7.2 mmol/L,5.60%,DEAD
NAGARAJ VEMPULI,241218283990,46,MALE,BREATHLESSNESS,CKD DCLD SHTN,180 mm Hg,110 mm Hg,180/110 mm Hg,91%,30 min⁻¹,130 mg/dL,97.8 °F,94 BPM,103 mg/dL,7.0 mg/dL,32 mg/L,1.2 mmol/L,0.7 mmol/L,41.67%,ALIVE
GIRIBABU,241216283391,51,MALE,CHEST PAIN,CKD T2DM SHTN,150 mm Hg,100 mm Hg,150/100 mm Hg,99%,22 min⁻¹,125 mg/dL,98.2 °F,63 BPM,64 mg/dL,4.6 mg/dL,124 mg/L,1.4 mmol/L,1.1 mmol/L,20.29%,ALIVE
AYYASWAMY,241206279973,74,MALE,DIALYSIS,CKD T2DM CLD,180 mm Hg,100 mm Hg,180/100 mm Hg,98%,21 min⁻¹,140 mg/dL,98.8 °F,97 BPM,105 mg/dL,4.4 mg/dL,76 mg/L,7.4 mmol/L,9.3 mmol/L,-25.68%,DEAD
MALLIGA,241207280316,54,FEMALE,BREATHLESSNESS,CKD T2DM SHTN,200 mm Hg,100 mm Hg,200/100 mm Hg,96%,21 min⁻¹,82 mg/dL,97.6 °F,95 BPM,49 mg/dL,2.3 mg/dL, 4 mg/L,4.8 mmol/L,2.2 mmol/L,54.17%,ALIVE
RAJASEKAR,241210281753,55,MALE,CHEST PAIN,CKD T2DM CAD,130 mm Hg,80 mm Hg,130/80 mm Hg,98%,24 min⁻¹,270 mg/dL,98.2 °F,97 BPM,73 mg/dL,3.9 mg/dL,4 mg/L,1.3 mmol/L,0.9 mmol/L,29.69%,ALIVE
MAHBOOB BASHA,241207280364,71,MALE,BREATHLESSNESS,CKD T2DM,120 mm Hg,70 mm Hg,120/70 mm Hg,68%,20 min⁻¹,155 mg/dL,97.1 °F,96 BPM,120 mg/dL,5.4 mg/dL,66 mg/L,1.3 mmol/L,2.7 mmol/L,-103.01%,DEAD
SUBRAMANI NEELAKANDAN,241128277787,77,MALE,BREATHLESSNESS ABDOMINAL PAIN,CKD T2DM CAD,180 mm Hg,70 mm Hg,180/70 mm Hg,91%,26 min⁻¹,155 mg/dL,97.2 °F,110 BPM,160 mg/dL,7.3 mg/dL,55 mg/L,1.6 mmol/L,1.3 mmol/L,18.75%,ALIVE
ARUMUGAM ,241127277450,60,MALE,GIDDINESS,CKD T2DM,150 mm Hg,100 mm Hg,150/100 mm Hg,100%,22 min⁻¹,54 mg/dL,98.0 °F,100 BPM,67 mg/dL,3.0 mg/dL,24 mg/L,2.5 mmol/L,1.8 mmol/L,28.00%,ALIVE
THANIGAIMALAI,241125276635,47,MALE,ALTERED SENSORIUM,CKD T2DM,140 mm Hg,90 mm Hg,140/90 mm Hg,97%,14 min⁻¹,303 mg/dL,98.1 °F,82 BPM,146 mg/dL,7.3 mg/dL,222 mg/L,2.4 mmol/L,1.2 mmol/L,50.00%,ALIVE
RAMESH,241117273480,52,MALE,VOMITING GIDDINESS,DCLD CKD T2DM,120 mm Hg,80 mm Hg,120/80 mm Hg,100%,22 min⁻¹,168 mg/dL,97.9 °F,91 BPM,135 mg/dL,3.2 mg/dL,13 mg/L,7.7 mmol/L,2.5 mmol/L,67.53%,ALIVE
RAMACHANDRA RAJU,241007259458,67,MALE,DECREASED URINE OUTPUT,CKD APE SHTN,140 mm Hg,100 mm Hg,140/100 mm Hg,80%,24 min⁻¹,77 mg/dL,99.6 °F,100 BPM,230 mg/dL,9.9 mg/dL,238 mg/L,4.3 mmol/L,6.4 mmol/L,4.80%,DEAD
PERUMAL RAJI,241013261039,64,MALE,BREATHLESSNESS COUGH,CKD SHTN T2DM CAD,150 mm Hg,100 mm Hg,150/100 mm Hg,85%,24 min⁻¹,134 mg/dL,98.1 °F,93 BPM,35 mg/dL,2.3 mg/dL,4 mg/L,2.1 mmol/L,0.7 mmol/L,66.67%,ALIVE
SHANMUGAN SUNDARAM,241109269609,59,MALE,BREATHLESSNESS,CKD SHTN,170 mm Hg,100 mm Hg,170/100 mm Hg,98%,24 min⁻¹,111 mg/dL,98.0 °F,72 BPM,83 mg/dL,4.5 mg/dL,4 mg/L,1.9 mmol/L,1.2 mmol/L,36.84%,ALIVE
NAGARATHINAM MUNUSWAMY,240812234251,63,MALE,GIDDINESS,CKD,110 mm Hg,60 mm Hg,110/60 mm Hg,98%,22 min⁻¹,191 mg/dL,96.7 °F,112 BPM,39 mg/dL,4.0 mg/dL,34 mg/L,3.7 mmol/L,0.8 mmol/L,78.32%,ALIVE
VINAYAGI,241015261664,56,FEMALE,BREATHLESSNESS,CKD SHTN T2DM SEIZURE DISORDER,140 mm Hg,70 mm Hg,140/70 mm Hg,65%,34 min⁻¹,217 mg/dL,97.4 °F,108 BPM,76 mg/dL,2.8 mg/dL,12 mg/L,1.7 mmol/L,1.0 mmol/L,41.18%,ALIVE
KAMSALA,241002257755,74,FEMALE,BREATHLESSNESS FACE PUFFINESS,CKD CAD T2DM APE ,110 mm Hg,60 mm Hg,110/60 mm Hg,84%,23 min⁻¹,304 mg/dL,96.2 °F,54 BPM,57 mg/dL,1.8 mg/dL,4 mg/L,10.8 mmol/L,8.7 mmol/L,19.91%,ALIVE
SANTHOSH KUMAR,240922252824,28,MALE,FEVER ,CEREBRAL PALSY CKD APE,80 mm Hg,60 mm Hg,80/60 mm Hg,85%,45 min⁻¹,86 mg/dL,98.1 °F,140 BPM,114 mg/dL,3.8 mg/dL,334 mg/L,2.1 mmol/L,4.5 mmol/L,1.43%,ALIVE
KASTHURI,231028145682,70,FEMALE,FEVER VOMITING LOOSE STOOLS,CKD SHTN HYPOTHYROIDISM HEPATOMEGALY,80 mm Hg,50 mm Hg,80/50 mm Hg,98%,24 min⁻¹,104 mg/dL,97.2 °F,98 BPM,98  mg/dL,3.5 mg/dL,420 mg/L,1.7 mmol/L,4.6 mmol/L,-26.80%,DEAD
NAGAMMAL,240911245108,46,FEMALE,GIDDINESS VOMITING,T2DM SHTN DYSLIPIDEMIA,180 mm Hg,110 mm Hg,180/110 mm Hg,93%,22 min⁻¹,182 mg/dL,99.0 °F,102 BPM,212 mg/dL,17.2 mg/dL,33 mg/L,1.8 mmol/L,1.2 mmol/L,34.78%,ALIVE
LOGANATHAN ,240911245091,45,MALE,FEVER DECREASED URINE OUTPUT,CAD SHTN,150 mm Hg,80 mm Hg,150/80 mm Hg,100%,33 min⁻¹,155 mg/dL,99.3 °F,133 BPM,170 mg/dL,12.2 mg/dL,397 mg/L,2.5 mmol/L,2.1 mmol/L,52.00%,ALIVE
PAARI,240911244840,59,MALE,B/L LIMB WEAKNESS SWELLING,DM CKD SHTN,150 mm Hg,90 mm Hg,150/90 mm Hg,93%,25 min⁻¹,508 mg/dL,97.2 °F,57 BPM,44 mg/dL,2.0 mg/dL,22 mg/L,2.7 mmol/L,0.9 mmol/L,66.67%,ALIVE
KUPPAN,231224162427,63,MALE,B/L UPPER LIMB WEAKNESS,CKD SHTN CVA,150 mm Hg,100 mm Hg,150/100 mm Hg,100%,20 min⁻¹,152 mg/dL,97.6 °F,60 BPM,250 mg/dL,5.9 mg/dL,86 mg/L,2.1 mmol/L,3.4 mmol/L,-61.90%,DEAD
VIJAYALAKSHMI,240907242701,73,FEMALE,CHEST PAIN BREATHLESSNESS,T2DM SHTN CAD CKD,180 mm Hg,120 mm Hg,180/120 mm Hg,85%,30 min⁻¹,340 mg/dL,98.6 °F,122 BPM,54 mg/dL,2.0 mg/dL,22 mg/L,4.3 mmol/L,2.0 mmol/L,53.49%,ALIVE
GOPAL NAYAKAR KAMARAJAN,240904241576,64,MALE,GIDDINESS VOMITING,CAD SHTN T2DM,170 mm Hg,90 mm Hg,170/90 mm Hg,86%,24 min⁻¹,182 mg/dL,98.0 °F,88 BPM,63 mg/dL,2.5 mg/dL,45 mg/L,3.7 mmol/L,2.9 mmol/L,21.62%,ALIVE
SAYEERA BEGUM,240831240468,66,FEMALE,BREATHLESSNESS,T2DM SHTN CKD,160 mm Hg,100 mm Hg,160/100 mm Hg,90%,23 min⁻¹,226 mg/dL,97.3 °F,101 BPM,131 mg/dL,7.6 mg/dL,15 mg/L,1.5 mmol/L,0.8 mmol/L,45.58%,ALIVE
AJAYA KUMAR YADAVA,231028245360,55,MALE,ABDOMINAL DISTENSION VOMITING,CKD DCLD SHTN T2DM,120 mm Hg,90 mm Hg,120/90 mm Hg,100%,33 min⁻¹,236 mg/dL,98.1 °F,86 BPM,126 mg/dL,2.4 mg/dL,70 mg/L,2.1 mmol/L,5.2 mmol/L,-147.62%,DEAD
MAHENTHRAN,240831240184,53,FEMALE,GIDDINESS VOMITING, CKD CAD SHTN T2DM HF,160 mm Hg,100 mm Hg,160/100 mm Hg,96%,18 min⁻¹,375 mg/dL,98.4 °F,92 BPM,41 mg/dL,1.5 mg/dL,77 mg/L,3.8 mmol/L,2.1 mmol/L,44.74%,ALIVE
RAVI,240829239629,57,MALE,BREATHLESSNESS,DM SHTN CKD CAD,160 mm Hg,110 mm Hg,160/110 mm Hg,84%,22 min⁻¹,228 mg/dL,98.2 °F,120 BPM,56 mg/dL,2.9 mg/dL,299 mg/L,0.8 mmol/L,0.4 mmol/L,50.00%,ALIVE
LALITHA,240813234351,64,FEMALE,DECREASED URINE OUTPUT BREATHLESSNESS,CKD ,160 mm Hg,80 mm Hg,160/80 mm Hg,100%,24 min⁻¹,98 mg/dL,98.4 °F,98 BPM,84 mg/dL,9.6 mg/dL,27 mg/L,9.2 mmol/L,7.2 mmol/L,21.74%,ALIVE
MALLIGA NATESAN,240809233063,67,FEMALE,BREATHLESSNESS,CAD CKD T2DM SHTN CA BREAST,140 mm Hg,100 mm Hg,140/100 mm Hg,82%,27 min⁻¹,306 mg/dL,97.6 °F,84 BPM,88 mg/dL,2.9 mg/dL,35 mg/L,2.4 mmol/L,1.3 mmol/L,42.98%,ALIVE
JAYANTHI SLEVARAJ,240807232656,58,FEMALE,BREATHLESSNESS,T2DM SHTN HYPOTHYROIDISM,140 mm Hg,80 mm Hg,140/80 mm Hg,85%,27 min⁻¹,245 mg/dL,97.5 °F,112 BPM,72 mg/dL,4.3 mg/dL,348 mg/L,4.3 mmol/L,2.6 mmol/L,39.53%,ALIVE
SALOMI,240727229194,52,FEMALE,BREATHLESSNESS,CKD SHTN T2DM,220 mm Hg,120 mm Hg,220/120 mm Hg,98%,21 min⁻¹,98 mg/dL,98.4 °F,102 BPM,103 mg/dL,8.7 mg/dL,112 mg/L,2.2 mmol/L,1.3 mmol/L,40.91%,ALIVE
DURGADEVI,240725228636,54,FEMALE,BREATHLESSNESS GIDDINESS,CKD T2DM HYPOTHYROIDISM ANEMIA,150 mm Hg,90 mm Hg,150/90 mm Hg,90%,24 min⁻¹,259 mg/dL,97.6 °F,88 BPM,138 mg/dL,9.2 mg/dL,135 mg/L,2.3 mmol/L,1.0 mmol/L,57.39%,ALIVE
ARJUN,240720226954,30,MALE,COUGH BREATHLESSNESS VOMITING,T2DM SHTN ,210 mm Hg,110 mm Hg,210/110 mm Hg,75%,32 min⁻¹,151 mg/dL,96.8 °F,100 BPM,197 mg/dL,15.2 mg/dL,39 mg/L,2.5 mmol/L,2.0 mmol/L,20.00%,ALIVE
MEENATCHI,240714224929,74,FEMALE,BREATHLESSNESS,CKD SHTN T2DM ,120 mm Hg,80 mm Hg,120/80 mm Hg,64%,30 min⁻¹,125 mg/dL,98.2 °F,62 BPM,100 mg/dL,9.5 mg/dL,8 mg/L,19.0 mmol/L,20.0 mmol/L,5.55%,DEAD
ANHALATCHI NAGAPPAN,240705222222,59,FEMALE, ABDOMINAL PAIN VOMITING,T2DM SHTN CKD ,100 mm Hg,60 mm Hg,100/60 mm Hg,95%,20 min⁻¹,299 mg/dL,98.0 °F,80 BPM,53 mg/dL,2.3 mg/dL,178 mg/L,1.8 mmol/L,1.4 mmol/L,23.08%,ALIVE
PUSHPA,240709223296,63,FEMALE,BREATHLESSNESS,T2DM SHTN CKD CAD,200 mm Hg,120 mm Hg,200/120 mm Hg,95%,28 min⁻¹,234 mg/dL,98.6 °F,114 BPM,123 mg/dL,5.2 mg/dL,16 mg/L,0.9 mmol/L,0.8 mmol/L,11.11%,ALIVE
HASINA,240704221852,69,FEMALE,LEG SWELLING BREATHLESSNESS,T2DM SHTN,150 mm Hg,70 mm Hg,150/70 mm Hg,83%,26 min⁻¹,164 mg/dL,98.6 °F,82 BPM,82 mg/dL,2.6 mg/dL,17 mg/L,1.1 mmol/L,0.9 mmol/L,18.18%,ALIVE
GLADYS JOSEPH,240703221437,83,FEMALE,LETHARGY DECREASED RESPONSE,CAD CKD SHTN HYPOTHYROIDISM ANEMIA,180 mm Hg,100 mm Hg,180/100 mm Hg,99%,20 min⁻¹,49 mg/dL,98.2 °F,80 BPM,166 mg/dL,6.8 mg/dL,61 mg/L,1.2 mmol/L,0.8 mmol/L,31.62%,ALIVE
VARDHARAJAN,240601211772,58,MALE,FALL FROM BED,CAD SHTN PARKINSONS DISEASE,80 mm Hg,60 mm Hg,80/60 mm Hg,99%,36 min⁻¹,158 mg/dL,101.5 °F,105 BPM,111 mg/dL,3.0 mg/dL,8 mg/L,1.3 mmol/L,2.3 mmol/L,7.60%,DEAD
PARAMESHWARAN,240531211348,77,MALE,NUMBNESS B/L UPPER LOWER LIMB, CKD SHTN T2DM CAD,150 mm Hg,100 mm Hg,150/100 mm Hg,98%,24 min⁻¹,174 mg/dL,97.4 °F,94 BPM,84 mg/dL,2.5 mg/dL,25 mg/L,3.8 mmol/L,2.1 mmol/L,44.88%,ALIVE
GANESAN,240528210216,64,MALE,BREATHLESSNESS,T2DM SHTN CKD ,160 mm Hg,90 mm Hg,160/90 mm Hg,93%,30 min⁻¹,119 mg/dL,97.5 °F,95 BPM,74 mg/dL,5.6 mg/dL,24 mg/L,1.6 mmol/L,1.4 mmol/L,12.50%,ALIVE
NATARAJAN,230807117024,56,MALE,NECK SWELLING,CKD TB SKTN CAD,140 mm Hg,90 mm Hg,140/90 mm Hg,99%,20 min⁻¹,380 mg/dL,99.3 °F,112 BPM,211 mg/dL,3.4 mg/dL,112 mg/L,1.4 mmol/L,1.6 mmol/L,-14.29%,DEAD
SARALADEVI,250505331418,47,FEMALE,BREATHLESSNESS,THYROID SHTN CKD,100 mm Hg,80 mm Hg,100/80 mm Hg,99%,24 min⁻¹,143 mg/dL,98.2 °F,143 BPM,54 mg/dL,0.8 mg/dL,38 mg/L,13.1 mmol/L,1.5 mmol/L,88.55%,ALIVE
MUNUSWAMY,250501330250,58,MALE,BREATHLESSNESS,SHTN CKD CAD,160 mm Hg,60 mm Hg,160/60 mm Hg,100%,24 min⁻¹,148 mg/dL,97.6 °F,93 BPM,339 mg/dL,26.7 mg/dL,232 mg/L,0.8 mmol/L,0.6 mmol/L,25.00%,ALIVE
GUBENDIRAN,240718226372,58,MALE,BREATHLESSNESS,SHTN CKD CAD,120 mm Hg,80 mm Hg,120/80 mm Hg,98%,26 min⁻¹,134 mg/dL,97.1 °F,78 BPM,193 mg/dL,16.3 mg/dL,30 mg/L,8.1 mmol/L,1.2 mmol/L,85.19%,ALIVE
PUNITHA SAMBATH,230823121809,61,FEMALE,GENERALISED FATIGUE,SHTN T2DM HF HYPOTHYROIDISM APE,130 mm Hg,90 mm Hg,130/90 mm Hg,99%,20 min⁻¹,190 mg/dL,96.5 °F,84 BPM,143 mg/dL,4.7 mg/dL,23 mg/L,2.2 mmol/L,2.5 mmol/L,-13.64%,DEAD
MAARTHAL,250501330270,55,FEMALE,BREATHLESSNESS FEVER,SHTN CKD CAD,130 mm Hg,70 mm Hg,130/70 mm Hg,99%,22 min⁻¹,140 mg/dL,97.1 °F,98 BPM,66 mg/dL,2.6 mg/dL,22 mg/L,8.5 mmol/L,6.3 mmol/L,25.88%,ALIVE
KASI,250517335421,65,FEMALE,BREATHLESSNESS PEDEAL EDEMA,SHTN CKD APE,110 mm Hg,70 mm Hg,110/70 mm Hg,100%,24 min⁻¹,151 mg/dL,97.2 °F,60 BPM,148 mg/dL,2.9 mg/dL,35 mg/L,7.8 mmol/L,1.1 mmol/L,85.90%,ALIVE
PRASANTH,241209281243,26,MALE,BREATHLESSNESS,SHTN CKD,110 mm Hg,70 mm Hg,110/70 mm Hg,98%,32 min⁻¹,125 mg/dL,98.1 °F,80 BPM,119 mg/dL,8.5 mg/dL,120 mg/L,1.5 mmol/L,1.1 mmol/L,26.67%,ALIVE
KASTHURI,250504331017,49,FEMALE,BREATHLESSNESS CHEST PAIN,SHTN T2DM CKD,130 mm Hg,80 mm Hg,130/80 mm Hg,98%,22 min⁻¹,201 mg/dL,97.4 °F,88 BPM,107 mg/dL,6.7 mg/dL,19 mg/L,1.7 mmol/L,1.2 mmol/L,29.82%,ALIVE
PROMODHA KUMARI,250519335853,70,FEMALE,BREATHLESSNESS FEVER,SHTN T2DM CKD,140 mm Hg,80 mm Hg,140/80 mm Hg,89%,26 min⁻¹,152 mg/dL,99.4 °F,78 BPM,90 mg/dL,3.8 mg/dL,345 mg/L,2.3 mmol/L,1.4 mmol/L,39.13%,ALIVE
KARUNAKARAN,2210105060,70,MALE,BREATHLESSNESS,SHTN T2DM CAD,150 mm Hg,80 mm Hg,150/80 mm Hg,85%,27 min⁻¹,186 mg/dL,98.3 °F,90 BPM,288 mg/dL,3.8 mg/dL,248 mg/L,2.2 mmol/L,4.1 mmol/L,8.90%,DEAD
GUNA,22102813258,57,MALE, BREATHLESSNESS FEVER,SHTN CKD,180 mm Hg,100 mm Hg,180/100 mm Hg,83%,40 min⁻¹,151 mg/dL,102.0 °F,151 BPM,102 mg/dL,4.3 mg/dL,323 mg/L,3.2 mmol/L,4.1 mmol/L,5.20%,DEAD
RUKUMANI,22111119149,68,FEMALE,BREATHLESSNESS FEVER,SHTN T2DM,110 mm Hg,60 mm Hg,110/60 mm Hg,100%,19 min⁻¹,267 mg/dL,99.3 °F,70 BPM,138 mg/dL,2.5 mg/dL,224 mg/L,1.6 mmol/L,2.5 mmol/L,5.60%,DEAD
MAYANBATHBEE ALIYAR,22111520325,62,FEMALE,BREATHLESSNESS CHEST PAIN,SHTN T2DM PE CKD CAD,160 mm Hg,90 mm Hg,160/90 mm Hg,82%,28 min⁻¹,261 mg/dL,96.2 °F,102 BPM,144 mg/dL,7.7 mg/dL,11 mg/L,2.1 mmol/L,3.2 mmol/L,-52.38%,DEAD
KARTHICK,22111520562,32,MALE,B/L LIMB WEAKNESS SWELLING ,T2DM CKD IDA SHTN,170 mm Hg,100 mm Hg,170/100 mm Hg,97%,22 min⁻¹,172 mg/dL,100.1 °F,123 BPM,57 mg/dL,2.0 mg/dL,34 mg/L,3.2 mmol/L,2.1 mmol/L,34.38%,ALIVE
ANNAMALAI GOVIDAN,23031267567,54,MALE,LOOSE STOOLS ABDOMINAL PAIN,T2DM HYPOTHYROID CKD,200 mm Hg,110 mm Hg,200/110 mm Hg,100%,26 min⁻¹,60 mg/dL,97.2 °F,54 BPM,200 mg/dL,5.7 mg/dL,4 mg/L,3.2 mmol/L,3.5 mmol/L,9.21,DEAD
MUTHUKUMAR,22122737627,54,MALE,BREATHLESSNESS FOOT ULCER,SHTN CKD,140 mm Hg,70 mm Hg,140/70 mm Hg,98%,22 min⁻¹,231 mg/dL,98.3 °F,90 BPM,102 mg/dL,9.4 mg/dL,42 mg/L,6.5 mmol/L,7.5 mmol/L,-15.38%,DEAD
RAJENDHIRAN,231121152560,66,MALE,SCROTAL SWELLING ,SHTN CKD T2DM,170 mm Hg,110 mm Hg,170/110 mm Hg,100%,19 min⁻¹,506 mg/dL,98.2 °F,116 BPM,179 mg/dL,6.2 mg/dL,238 mg/L,2.1 mmol/L,3.4 mmol/L,-61.90%,DEAD
SUBRAMANIYAM PONNUSWAMY,22122737683,71,MALE,BREATHLESSNESS,SHTN T2DM CKD CAD,180 mm Hg,90 mm Hg,180/90 mm Hg,97%,22 min⁻¹,251 mg/dL,97.7 °F,120 BPM,100 mg/dL,2.4 mg/dL,86 mg/L,1.2 mmol/L,3.5 mmol/L,-191.67%,DEAD
ANTHONYAMMAL,22122938233,73,FEMALE,VOMITING SLURRING OF SPEECH,CA BREAST SHTN T2DM ANEMIA,120 mm Hg,80 mm Hg,120/80 mm Hg,99%,18 min⁻¹,53 mg/dL,97.4 °F,98 BPM,69 mg/dL,2.2 mg/dL,224 mg/L,1.1 mmol/L,1.6 mmol/L,-45.45%,DEAD
KABILAN,22123038771,22,MALE,BREATHLESSNESS PEDEAL EDEMA,SHTN CKD,140 mm Hg,80 mm Hg,140/80 mm Hg,100%,23 min⁻¹,102 mg/dL,95.2 °F,84 BPM,54 mg/dL,2.5 mg/dL,53 mg/L,1.2 mmol/L,2.3 mmol/L,9.10%,DEAD
PERUMAL SUBRAMANI,240105166005,50,MALE,B/L LOWER LIMB SWELLING,T2DM SHTN CKD,150 mm Hg,90 mm Hg,150/90 mm Hg,76%,30 min⁻¹,409 mg/dL,100.0 °F,114 BPM,109 mg/dL,8.5 mg/dL,221 mg/L,1.9 mmol/L,1.4 mmol/L,26.32%,ALIVE
NAGARAJAN,240127171953,43,MALE,BREATHLESSNESS,T2DM CKD SHTN HF ,130 mm Hg,80 mm Hg,130/80 mm Hg,100%,22 min⁻¹,324 mg/dL,97.5 °F,76 BPM,107 mg/dL,7.8 mg/dL,60 mg/L,0.7 mmol/L,1.2 mmol/L,7.14%,DEAD
KANTHAL AMMAL,240129172139,63,FEMALE,LOOSE STOOLS,CKD DM SHTN CAD,140 mm Hg,80 mm Hg,140/80 mm Hg,100%,24 min⁻¹,40 mg/dL,98.2 °F,94 BPM,74 mg/dL,3.5 mg/dL,231 mg/L,2.4 mmol/L,1.7 mmol/L,29.17%,ALIVE
DHANALAKSHMI,22110114624,58,FEMALE,ABDOMINAL PAIN,T2DM CKD,100 mm Hg,80 mm Hg,100/80 mm Hg,98%,24 min⁻¹,444 mg/dL,98.6 °F,84 BPM,93 mg/dL,3.5 mg/dL,33 mg/L,2.4 mmol/L,1.0 mmol/L,58.33%,ALIVE
DHANRAJ,22110616626,63,MALE,BREATHLESSNESS,T2DM CKD SHTN CAD PE,170 mm Hg,80 mm Hg,170/80 mm Hg,66%,28 min⁻¹,150 mg/dL,98.2 °F,80 BPM,86 mg/dL,6.2 mg/dL,89 mg/L,0.9 mmol/L,0.8 mmol/L,11.11%,ALIVE
ABDUL KAREEM,22112223927,57,MALE,DECREASED APPETITE,CKD SHTN CAD T2DM,100 mm Hg,80 mm Hg,100/80 mm Hg,100%,38 min⁻¹,170 mg/dL,101.4 °F,115 BPM,128 mg/dL,4.6 mg/dL,289 mg/L,5.3 mmol/L,2.1 mmol/L,60.38%,ALIVE
MAQBOOL KHADAR,23011945838,67,MALE,HEMATOCHEZIA,CAD CKD SHTN  ,130 mm Hg,70 mm Hg,130/70 mm Hg,100%,22 min⁻¹,103 mg/dL,98.3 °F,66 BPM,105 mg/dL,3.7 mg/dL,186 mg/L,1.3 mmol/L,2.0 mmol/L,-53.85%,DEAD
DEVAKI HARIKRISHNAN,23050985837,68,FEMALE,BREATHLESSNESS,CKD APE SHTN T2DM,200 mm Hg,90 mm Hg,200/90 mm Hg,85%,30 min⁻¹,152 mg/dL,98.6 °F,83 BPM,34 mg/dL,5.7 mg/dL,12 mg/L,1.2 mmol/L,2.1 mmol/L,-75.00%,DEAD
SANKAR GOPAL,23012146816,60,MALE,FEVER,T2DM SHTN CKD,160 mm Hg,80 mm Hg,160/80 mm Hg,89%,20 min⁻¹,118 mg/dL,100.9 °F,102 BPM,64 mg/dL,9.1 mg/dL,53 mg/L,1.4 mmol/L,0.8 mmol/L,42.86%,ALIVE
VASANTHA THIRUVENGADAM,2302085575,53,FEMALE,CHEST PAIN,CKD APE SHTN CAD,170 mm Hg,90 mm Hg,170/90 mm Hg,87%,22 min⁻¹,243 mg/dL,98.2 °F,90 BPM,114 mg/dL,7.1 mg/dL,231 mg/L,3.1 mmol/L,2.4 mmol/L,22.58%,ALIVE
RAMANI BAI PARTHASARATHY,23021558308,65,FEMALE,ABDOMINAL PAIN ,SHTN T2DM CKD CAD,150 mm Hg,80 mm Hg,150/80 mm Hg,100%,21 min⁻¹,227 mg/dL,96.0 °F,130 BPM,87 mg/dL,6.8 mg/dL,102 mg/L,3.2 mmol/L,3.4 mmol/L,6.25%,DEAD
VIJAYALAKSHMI,23030464465,55,FEMALE,GIDDINESS VOMITING,T2DM SHTN,180 mm Hg,70 mm Hg,180/70 mm Hg,98%,19 min⁻¹,230 mg/dL,97.3 °F,98 BPM,71 mg/dL,6.9 mg/dL,197 mg/L,1.6 mmol/L,2.5 mmol/L,-56.25%,DEAD
SAMBHU KUMAR,23030765490,29,MALE,FEVER BREATHLESSNESS,SHTN CKD,90 mm Hg,60 mm Hg,90/60 mm Hg,88%,36 min⁻¹,165 mg/dL,97.2 °F,106 BPM,282 mg/dL,27.7 mg/dL,234 mg/L,1.9 mmol/L,1.0 mmol/L,47.37%,ALIVE
UMADEVI RAMESH ,23052692346,44,FEMALE,ABDOMINAL PAIN CHEST PAIN,SHTN CKD T2DM,80 mm Hg,60 mm Hg,80/60 mm Hg,89%,29 min⁻¹,116 mg/dL,97.0 °F,114 BPM,151 mg/dL,7.5 mg/dL,255 mg/L,2.5 mmol/L,3.1 mmol/L,2.40%,DEAD
SASIDHARAN,23061498911,70,MALE,CHEST PAIN BREATHLESSNESS,CKD SHTN ACS ,180 mm Hg,110 mm Hg,180/110 mm Hg,96%,18 min⁻¹,132 mg/dL,97.2 °F,86 BPM,102 mg/dL,9.9 mg/dL,102 mg/L,1.3 mmol/L,2.3 mmol/L,-76.92%,DEAD
MURUGAN V ,230710107371,60,MALE,BREATHLESSNESS,CKD SHTN,140 mm Hg,70 mm Hg,140/70 mm Hg,89%,24 min⁻¹,133 mg/dL,97.6 °F,90 BPM,203 mg/dL,12.9 mg/dL,285 mg/L,1.4 mmol/L,2.4 mmol/L,-71.43%,DEAD
SARAVANAN,230818120368,51,MALE,SLIP AND FALL BACK PAIN,CKD CAD T2DM SHTN,130 mm Hg,90 mm Hg,130/90 mm Hg,99%,20 min⁻¹,167 mg/dL,98.1 °F,88 BPM,121 mg/dL,7.7 mg/dL,84 mg/L,3.1 mmol/L,4.5 mmol/L,-45.16%,DEAD
RAJAAMMAL,230825122789,80,FEMALE,BACK PAIN LOSS OF APPETITE,CKD SHTN,100 mm Hg,60 mm Hg,100/60 mm Hg,97%,20 min⁻¹,102 mg/dL,97.0 °F,99 BPM,57 mg/dL,1.7 mg/dL,56 mg/L,1.7 mmol/L,2.4 mmol/L,8.60%,DEAD
SUBBIRAMANI,230910127920,58,MALE,BREATHLESSNESS,CKD T2DM SHTN APE,120 mm Hg,80 mm Hg,120/80 mm Hg,98%,22 min⁻¹,200 mg/dL,98.2 °F,104 BPM,345 mg/dL,16.6 mg/dL,83 mg/L,1.4 mmol/L,1.6 mmol/L,-14.29%,DEAD"""

# Create DataFrame from embedded data
from io import StringIO
df = pd.read_csv(StringIO(csv_data))

# Clean data columns
df['INITIAL LACTATE (clean)'] = df['INITIAL LACTATE'].str.extract(r'([\d.]+)').astype(float)
df['LACTATE CLEARANCE (clean)'] = df['LACTATE CLEARANCE'].str.extract(r'([\d.]+)').astype(float)
df['REPEAT LACTATE (clean)'] = df['REPEAT LACTATE'].str.extract(r'([\d.]+)').astype(float)
try:
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    analysis_type = st.sidebar.selectbox(
    "Select Analysis",
    ["Overview", "Initial Lactate Analysis", "Lactate Clearance Analysis", "Repeat Lactate Analysis", "Age Analysis", "CAD Analysis", "SHTN+T2DM Analysis", "Unstable Hemodynamic Analysis", "Combined Analysis"]
)

    if analysis_type == "Overview":
        st.header("📊 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Patients", len(df))
        with col2:
            alive_count = len(df[df['CLINICAL OUTCOMES'] == 'ALIVE'])
            st.metric("Alive", alive_count)
        with col3:
            dead_count = len(df[df['CLINICAL OUTCOMES'] == 'DEAD'])
            st.metric("Dead", dead_count)
        with col4:
            survival_rate = (alive_count / len(df)) * 100
            st.metric("Survival Rate", f"{survival_rate:.1f}%")
        
        # Outcome distribution pie chart
        fig_pie = px.pie(
            values=[alive_count, dead_count],
            names=['ALIVE', 'DEAD'],
            title="Clinical Outcomes Distribution",
            color_discrete_map={'ALIVE': '#2E8B57', 'DEAD': '#DC143C'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Age distribution
        fig_age = px.histogram(
            df, x='AGE', color='CLINICAL OUTCOMES',
            title="Age Distribution by Clinical Outcome",
            nbins=20,
            color_discrete_map={'ALIVE': '#2E8B57', 'DEAD': '#DC143C'}
        )
        st.plotly_chart(fig_age, use_container_width=True)
    
    elif analysis_type == "Initial Lactate Analysis":
        st.header("🧪 Initial Lactate vs Clinical Outcomes")
        
        # Filter data
        filtered_df = df[['INITIAL LACTATE (clean)', 'CLINICAL OUTCOMES']].dropna()
        alive_group = filtered_df[filtered_df['CLINICAL OUTCOMES'] == 'ALIVE']['INITIAL LACTATE (clean)']
        dead_group = filtered_df[filtered_df['CLINICAL OUTCOMES'] == 'DEAD']['INITIAL LACTATE (clean)']
        
        # Mann-Whitney U Test
        u_stat, p_value = mannwhitneyu(alive_group, dead_group, alternative='two-sided')
        
        # Display test results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("U-Statistic", f"{u_stat:.2f}")
        with col2:
            st.metric("P-Value", f"{p_value:.4f}")
        with col3:
            significance = "Significant" if p_value < 0.05 else "Not Significant"
            st.metric("Result", significance)
        
        # Box plot
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=alive_group, name='ALIVE', marker_color='#2E8B57'))
        fig_box.add_trace(go.Box(y=dead_group, name='DEAD', marker_color='#DC143C'))
        fig_box.update_layout(
            title="Initial Lactate Distribution by Clinical Outcome",
            yaxis_title="Initial Lactate (mmol/L)",
            xaxis_title="Clinical Outcome"
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Violin plot
        fig_violin = go.Figure()
        fig_violin.add_trace(go.Violin(y=alive_group, name='ALIVE', box_visible=True, 
                                     line_color='#2E8B57', fillcolor='rgba(46,139,87,0.5)'))
        fig_violin.add_trace(go.Violin(y=dead_group, name='DEAD', box_visible=True,
                                     line_color='#DC143C', fillcolor='rgba(220,20,60,0.5)'))
        fig_violin.update_layout(
            title="Initial Lactate Distribution (Violin Plot)",
            yaxis_title="Initial Lactate (mmol/L)"
        )
        st.plotly_chart(fig_violin, use_container_width=True)
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**ALIVE Group**")
            st.write(f"Mean: {alive_group.mean():.2f} mmol/L")
            st.write(f"Median: {alive_group.median():.2f} mmol/L")
            st.write(f"Std Dev: {alive_group.std():.2f} mmol/L")
            st.write(f"Count: {len(alive_group)}")
        
        with col2:
            st.write("**DEAD Group**")
            st.write(f"Mean: {dead_group.mean():.2f} mmol/L")
            st.write(f"Median: {dead_group.median():.2f} mmol/L")
            st.write(f"Std Dev: {dead_group.std():.2f} mmol/L")
            st.write(f"Count: {len(dead_group)}")
    
    elif analysis_type == "Lactate Clearance Analysis":
        st.header("🔄 Lactate Clearance vs Clinical Outcomes")
        
        # Filter data (matching your approach)
        filtered_df = df[['LACTATE CLEARANCE (clean)', 'CLINICAL OUTCOMES']].dropna()
        alive_group = filtered_df[filtered_df['CLINICAL OUTCOMES'].str.upper() == 'ALIVE']['LACTATE CLEARANCE (clean)']
        dead_group = filtered_df[filtered_df['CLINICAL OUTCOMES'].str.upper() == 'DEAD']['LACTATE CLEARANCE (clean)']
        
        # Multiple Statistical Tests
        st.subheader("📊 Statistical Test Results")
        
        # 1. Mann-Whitney U Test
        u_stat, p_mw = mannwhitneyu(alive_group, dead_group, alternative='two-sided')
        
        # 2. Welch's t-test (unequal variances)
        t_stat, p_ttest = ttest_ind(alive_group, dead_group, equal_var=False)
        
        # 3. Kolmogorov-Smirnov test
        ks_stat, p_ks = ks_2samp(alive_group, dead_group)
        
        # 4. Permutation test
        from scipy.stats import permutation_test
        def stat_func(x, y):
            return np.mean(x) - np.mean(y)
        perm_test = permutation_test((alive_group, dead_group), stat_func, n_resamples=10000)
        p_perm = perm_test.pvalue
        
        # 5. Bootstrap test
        from scipy.stats import bootstrap
        boot_test = bootstrap((alive_group, dead_group), stat_func, n_resamples=10000)
        p_boot = boot_test.confidence_interval[0]
        
        # Create test results table
        test_results = {
            'Test': ['T-test', 'Mann-Whitney U', 'Kolmogorov-Smirnov', 'Permutation', 'Bootstrap'],
            'Statistic': [t_stat, u_stat, ks_stat, perm_test.statistic, boot_test.confidence_interval[0]],
            'P-Value': [p_ttest, p_mw, p_ks, p_perm, p_boot],
            'Significant': ['Yes' if p < 0.05 else 'No' for p in [p_ttest, p_mw, p_ks, p_perm, p_boot]]
        }
        
        results_df = pd.DataFrame(test_results)
        st.dataframe(results_df.round(4), use_container_width=True)
        
        # Highlight most significant test
        p_values = [p_ttest, p_mw, p_ks, p_perm, p_boot]
        min_p = min(p_values)
        best_test_idx = p_values.index(min_p)
        best_test = test_results['Test'][best_test_idx]
        st.success(f"🎯 Most significant result: **{best_test}** (p = {min_p:.4f})")
        
        # Box plot
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=alive_group, name='ALIVE', marker_color='#2E8B57'))
        fig_box.add_trace(go.Box(y=dead_group, name='DEAD', marker_color='#DC143C'))
        fig_box.update_layout(
            title="Lactate Clearance Distribution by Clinical Outcome",
            yaxis_title="Lactate Clearance (%)",
            xaxis_title="Clinical Outcome"
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Histogram
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=alive_group, name='ALIVE', opacity=0.7, 
                                      marker_color='#2E8B57', nbinsx=20))
        fig_hist.add_trace(go.Histogram(x=dead_group, name='DEAD', opacity=0.7,
                                      marker_color='#DC143C', nbinsx=20))
        fig_hist.update_layout(
            title="Lactate Clearance Distribution Histogram",
            xaxis_title="Lactate Clearance (%)",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**ALIVE Group**")
            st.write(f"Mean: {alive_group.mean():.2f}%")
            st.write(f"Median: {alive_group.median():.2f}%")
            st.write(f"Std Dev: {alive_group.std():.2f}%")
            st.write(f"Count: {len(alive_group)}")
        
        with col2:
            st.write("**DEAD Group**")
            st.write(f"Mean: {dead_group.mean():.2f}%")
            st.write(f"Median: {dead_group.median():.2f}%")
            st.write(f"Std Dev: {dead_group.std():.2f}%")
            st.write(f"Count: {len(dead_group)}")
    
    elif analysis_type == "Repeat Lactate Analysis":
        st.header("🔁 Repeat Lactate vs Clinical Outcomes")
        
        # Filter data
        filtered_df = df[['REPEAT LACTATE (clean)', 'CLINICAL OUTCOMES']].dropna()
        alive_group = filtered_df[filtered_df['CLINICAL OUTCOMES'].str.upper() == 'ALIVE']['REPEAT LACTATE (clean)']
        dead_group = filtered_df[filtered_df['CLINICAL OUTCOMES'].str.upper() == 'DEAD']['REPEAT LACTATE (clean)']
        
        # Multiple Statistical Tests
        st.subheader("📊 Statistical Test Results")
        
        # 1. Mann-Whitney U Test
        u_stat, p_mw = mannwhitneyu(alive_group, dead_group, alternative='two-sided')
        
        # 2. Welch's t-test (unequal variances)
        t_stat, p_ttest = ttest_ind(alive_group, dead_group, equal_var=False)
        
        # 3. Kolmogorov-Smirnov test
        ks_stat, p_ks = ks_2samp(alive_group, dead_group)
        
        # 4. Permutation test
        from scipy.stats import permutation_test
        def stat_func(x, y):
            return np.mean(x) - np.mean(y)
        perm_test = permutation_test((alive_group, dead_group), stat_func, n_resamples=10000)
        p_perm = perm_test.pvalue
        
        # 5. Bootstrap test
        from scipy.stats import bootstrap
        boot_test = bootstrap((alive_group, dead_group), stat_func, n_resamples=10000)
        p_boot = boot_test.confidence_interval[0]
        
        # Create test results table
        test_results = {
            'Test': ['T-test', 'Mann-Whitney U', 'Kolmogorov-Smirnov', 'Permutation', 'Bootstrap'],
            'Statistic': [t_stat, u_stat, ks_stat, perm_test.statistic, boot_test.confidence_interval[0]],
            'P-Value': [p_ttest, p_mw, p_ks, p_perm, p_boot],
            'Significant': ['Yes' if p < 0.05 else 'No' for p in [p_ttest, p_mw, p_ks, p_perm, p_boot]]
        }
        
        results_df = pd.DataFrame(test_results)
        st.dataframe(results_df.round(4), use_container_width=True)
        
        # Highlight most significant test
        p_values = [p_ttest, p_mw, p_ks, p_perm, p_boot]
        min_p = min(p_values)
        best_test_idx = p_values.index(min_p)
        best_test = test_results['Test'][best_test_idx]
        st.success(f"🎯 Most significant result: **{best_test}** (p = {min_p:.4f})")
        
        # Box plot
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=alive_group, name='ALIVE', marker_color='#2E8B57'))
        fig_box.add_trace(go.Box(y=dead_group, name='DEAD', marker_color='#DC143C'))
        fig_box.update_layout(
            title="Repeat Lactate Distribution by Clinical Outcome",
            yaxis_title="Repeat Lactate (mmol/L)",
            xaxis_title="Clinical Outcome"
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Histogram
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=alive_group, name='ALIVE', opacity=0.7, 
                                      marker_color='#2E8B57', nbinsx=20))
        fig_hist.add_trace(go.Histogram(x=dead_group, name='DEAD', opacity=0.7,
                                      marker_color='#DC143C', nbinsx=20))
        fig_hist.update_layout(
            title="Repeat Lactate Distribution Histogram",
            xaxis_title="Repeat Lactate (mmol/L)",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**ALIVE Group**")
            st.write(f"Mean: {alive_group.mean():.2f} mmol/L")
            st.write(f"Median: {alive_group.median():.2f} mmol/L")
            st.write(f"Std Dev: {alive_group.std():.2f} mmol/L")
            st.write(f"Count: {len(alive_group)}")
        
        with col2:
            st.write("**DEAD Group**")
            st.write(f"Mean: {dead_group.mean():.2f} mmol/L")
            st.write(f"Median: {dead_group.median():.2f} mmol/L")
            st.write(f"Std Dev: {dead_group.std():.2f} mmol/L")
            st.write(f"Count: {len(dead_group)}")
    
    elif analysis_type == "Age Analysis":
        st.header("👥 Age vs Clinical Outcomes")
        
        # Filter data
        filtered_df = df[['AGE', 'CLINICAL OUTCOMES']].dropna()
        alive_group = filtered_df[filtered_df['CLINICAL OUTCOMES'].str.upper() == 'ALIVE']['AGE']
        dead_group = filtered_df[filtered_df['CLINICAL OUTCOMES'].str.upper() == 'DEAD']['AGE']
        
        # Multiple Statistical Tests
        st.subheader("📊 Statistical Test Results")
        
        # 1. Mann-Whitney U Test
        u_stat, p_mw = mannwhitneyu(alive_group, dead_group, alternative='two-sided')
        
        # 2. Welch's t-test (unequal variances)
        t_stat, p_ttest = ttest_ind(alive_group, dead_group, equal_var=False)
        
        # 3. Kolmogorov-Smirnov test
        ks_stat, p_ks = ks_2samp(alive_group, dead_group)
        
        # 4. Permutation test
        from scipy.stats import permutation_test
        def stat_func(x, y):
            return np.mean(x) - np.mean(y)
        perm_test = permutation_test((alive_group, dead_group), stat_func, n_resamples=10000)
        p_perm = perm_test.pvalue
        
        # 5. Bootstrap test
        from scipy.stats import bootstrap
        boot_test = bootstrap((alive_group, dead_group), stat_func, n_resamples=10000)
        p_boot = boot_test.confidence_interval[0]
        
        # Create test results table
        test_results = {
            'Test': ['T-test', 'Mann-Whitney U', 'Kolmogorov-Smirnov', 'Permutation', 'Bootstrap'],
            'Statistic': [t_stat, u_stat, ks_stat, perm_test.statistic, boot_test.confidence_interval[0]],
            'P-Value': [p_ttest, p_mw, p_ks, p_perm, p_boot],
            'Significant': ['Yes' if p < 0.05 else 'No' for p in [p_ttest, p_mw, p_ks, p_perm, p_boot]]
        }
        
        results_df = pd.DataFrame(test_results)
        st.dataframe(results_df.round(4), use_container_width=True)
        
        # Highlight most significant test
        p_values = [p_ttest, p_mw, p_ks, p_perm, p_boot]
        min_p = min(p_values)
        best_test_idx = p_values.index(min_p)
        best_test = test_results['Test'][best_test_idx]
        st.success(f"🎯 Most significant result: **{best_test}** (p = {min_p:.4f})")
        
        # Box plot
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=alive_group, name='ALIVE', marker_color='#2E8B57'))
        fig_box.add_trace(go.Box(y=dead_group, name='DEAD', marker_color='#DC143C'))
        fig_box.update_layout(
            title="Age Distribution by Clinical Outcome",
            yaxis_title="Age (years)",
            xaxis_title="Clinical Outcome"
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Histogram
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=alive_group, name='ALIVE', opacity=0.7, 
                                      marker_color='#2E8B57', nbinsx=20))
        fig_hist.add_trace(go.Histogram(x=dead_group, name='DEAD', opacity=0.7,
                                      marker_color='#DC143C', nbinsx=20))
        fig_hist.update_layout(
            title="Age Distribution Histogram",
            xaxis_title="Age (years)",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**ALIVE Group**")
            st.write(f"Mean: {alive_group.mean():.1f} years")
            st.write(f"Median: {alive_group.median():.1f} years")
            st.write(f"Std Dev: {alive_group.std():.1f} years")
            st.write(f"Count: {len(alive_group)}")
        
        with col2:
            st.write("**DEAD Group**")
            st.write(f"Mean: {dead_group.mean():.1f} years")
            st.write(f"Median: {dead_group.median():.1f} years")
            st.write(f"Std Dev: {dead_group.std():.1f} years")
            st.write(f"Count: {len(dead_group)}")
    
    elif analysis_type == "CAD Analysis":
        st.header("❤️ CAD vs Clinical Outcomes")
        
        # Filter data for CAD analysis
        filtered_df = df[['K/C/O', 'CLINICAL OUTCOMES']].dropna()
        
        # Check if CAD is present in the K/C/O string
        filtered_df['has_CAD'] = filtered_df['K/C/O'].str.contains('CAD', case=False, na=False)
        
        cad_group = filtered_df[filtered_df['has_CAD'] == True]['CLINICAL OUTCOMES']
        no_cad_group = filtered_df[filtered_df['has_CAD'] == False]['CLINICAL OUTCOMES']
        
        # Create contingency table
        cad_alive = len(cad_group[cad_group.str.upper() == 'ALIVE'])
        cad_dead = len(cad_group[cad_group.str.upper() == 'DEAD'])
        no_cad_alive = len(no_cad_group[no_cad_group.str.upper() == 'ALIVE'])
        no_cad_dead = len(no_cad_group[no_cad_group.str.upper() == 'DEAD'])
        
        contingency_table = [[cad_alive, cad_dead], [no_cad_alive, no_cad_dead]]
        
        # Use Fisher's exact test if any cell has count < 5, otherwise chi-square
        if min(cad_alive, cad_dead, no_cad_alive, no_cad_dead) < 5:
            from scipy.stats import fisher_exact
            odds_ratio, p_chi2 = fisher_exact(contingency_table)
            chi2_stat = odds_ratio
            test_name = "Fisher's Exact Test"
        else:
            chi2_stat, p_chi2, dof, expected = chi2_contingency(contingency_table)
            test_name = "Chi-square Test"
        
        # Display test results
        st.subheader("📊 Statistical Test Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{test_name} Statistic", f"{chi2_stat:.4f}")
        with col2:
            st.metric("P-Value", f"{p_chi2:.4f}")
        with col3:
            significance = "Significant" if p_chi2 < 0.05 else "Not Significant"
            st.metric("Result", significance)
        
        # Contingency table display
        st.subheader("📋 Contingency Table")
        contingency_df = pd.DataFrame({
            'CAD': [cad_alive, cad_dead],
            'No CAD': [no_cad_alive, no_cad_dead]
        }, index=['ALIVE', 'DEAD'])
        st.dataframe(contingency_df, use_container_width=True)
        
        # Stacked bar chart
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name='ALIVE', x=['CAD', 'No CAD'], y=[cad_alive, no_cad_alive], marker_color='#2E8B57'))
        fig_bar.add_trace(go.Bar(name='DEAD', x=['CAD', 'No CAD'], y=[cad_dead, no_cad_dead], marker_color='#DC143C'))
        fig_bar.update_layout(
            title="Clinical Outcomes by CAD Status",
            xaxis_title="CAD Status",
            yaxis_title="Count",
            barmode='stack'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Survival rates
        st.subheader("📈 Survival Rates")
        col1, col2 = st.columns(2)
        
        with col1:
            cad_total = cad_alive + cad_dead
            cad_survival_rate = (cad_alive / cad_total * 100) if cad_total > 0 else 0
            st.write("**CAD Patients**")
            st.write(f"Total: {cad_total}")
            st.write(f"Alive: {cad_alive}")
            st.write(f"Dead: {cad_dead}")
            st.write(f"Survival Rate: {cad_survival_rate:.1f}%")
        
        with col2:
            no_cad_total = no_cad_alive + no_cad_dead
            no_cad_survival_rate = (no_cad_alive / no_cad_total * 100) if no_cad_total > 0 else 0
            st.write("**No CAD Patients**")
            st.write(f"Total: {no_cad_total}")
            st.write(f"Alive: {no_cad_alive}")
            st.write(f"Dead: {no_cad_dead}")
            st.write(f"Survival Rate: {no_cad_survival_rate:.1f}%")
    
    elif analysis_type == "SHTN+T2DM Analysis":
        st.header("💔 SHTN+T2DM vs Clinical Outcomes")
        
        # Filter data for SHTN+T2DM analysis
        filtered_df = df[['K/C/O', 'CLINICAL OUTCOMES']].dropna()
        
        # Check if both SHTN and T2DM are present in the K/C/O string
        filtered_df['has_SHTN_T2DM'] = (filtered_df['K/C/O'].str.contains('SHTN', case=False, na=False) | 
                                        filtered_df['K/C/O'].str.contains('T2DM', case=False, na=False))
        
        shtn_t2dm_group = filtered_df[filtered_df['has_SHTN_T2DM'] == True]['CLINICAL OUTCOMES']
        no_shtn_t2dm_group = filtered_df[filtered_df['has_SHTN_T2DM'] == False]['CLINICAL OUTCOMES']
        
        # Create contingency table
        shtn_t2dm_alive = len(shtn_t2dm_group[shtn_t2dm_group.str.upper() == 'ALIVE'])
        shtn_t2dm_dead = len(shtn_t2dm_group[shtn_t2dm_group.str.upper() == 'DEAD'])
        no_shtn_t2dm_alive = len(no_shtn_t2dm_group[no_shtn_t2dm_group.str.upper() == 'ALIVE'])
        no_shtn_t2dm_dead = len(no_shtn_t2dm_group[no_shtn_t2dm_group.str.upper() == 'DEAD'])
        
        contingency_table = [[shtn_t2dm_alive, shtn_t2dm_dead], [no_shtn_t2dm_alive, no_shtn_t2dm_dead]]
        
        # Use Fisher's exact test if any cell has count < 5, otherwise chi-square
        if min(shtn_t2dm_alive, shtn_t2dm_dead, no_shtn_t2dm_alive, no_shtn_t2dm_dead) < 5:
            from scipy.stats import fisher_exact
            odds_ratio, p_chi2 = fisher_exact(contingency_table)
            chi2_stat = odds_ratio
            test_name = "Fisher's Exact Test"
        else:
            chi2_stat, p_chi2, dof, expected = chi2_contingency(contingency_table)
            test_name = "Chi-square Test"
        
        # Display test results
        st.subheader("📊 Statistical Test Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{test_name} Statistic", f"{chi2_stat:.4f}")
        with col2:
            st.metric("P-Value", f"{p_chi2:.4f}")
        with col3:
            significance = "Significant" if p_chi2 < 0.05 else "Not Significant"
            st.metric("Result", significance)
        
        # Contingency table display
        st.subheader("📋 Contingency Table")
        contingency_df = pd.DataFrame({
            'SHTN+T2DM': [shtn_t2dm_alive, shtn_t2dm_dead],
            'Others': [no_shtn_t2dm_alive, no_shtn_t2dm_dead]
        }, index=['ALIVE', 'DEAD'])
        st.dataframe(contingency_df, use_container_width=True)
        
        # Stacked bar chart
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name='ALIVE', x=['SHTN+T2DM', 'Others'], y=[shtn_t2dm_alive, no_shtn_t2dm_alive], marker_color='#2E8B57'))
        fig_bar.add_trace(go.Bar(name='DEAD', x=['SHTN+T2DM', 'Others'], y=[shtn_t2dm_dead, no_shtn_t2dm_dead], marker_color='#DC143C'))
        fig_bar.update_layout(
            title="Clinical Outcomes by SHTN+T2DM Status",
            xaxis_title="Patient Group",
            yaxis_title="Count",
            barmode='stack'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Survival rates
        st.subheader("📈 Survival Rates")
        col1, col2 = st.columns(2)
        
        with col1:
            shtn_t2dm_total = shtn_t2dm_alive + shtn_t2dm_dead
            shtn_t2dm_survival_rate = (shtn_t2dm_alive / shtn_t2dm_total * 100) if shtn_t2dm_total > 0 else 0
            st.write("**SHTN+T2DM Patients**")
            st.write(f"Total: {shtn_t2dm_total}")
            st.write(f"Alive: {shtn_t2dm_alive}")
            st.write(f"Dead: {shtn_t2dm_dead}")
            st.write(f"Survival Rate: {shtn_t2dm_survival_rate:.1f}%")
        
        with col2:
            no_shtn_t2dm_total = no_shtn_t2dm_alive + no_shtn_t2dm_dead
            no_shtn_t2dm_survival_rate = (no_shtn_t2dm_alive / no_shtn_t2dm_total * 100) if no_shtn_t2dm_total > 0 else 0
            st.write("**Other Patients**")
            st.write(f"Total: {no_shtn_t2dm_total}")
            st.write(f"Alive: {no_shtn_t2dm_alive}")
            st.write(f"Dead: {no_shtn_t2dm_dead}")
            st.write(f"Survival Rate: {no_shtn_t2dm_survival_rate:.1f}%")
    
    elif analysis_type == "Unstable Hemodynamic Analysis":
        st.header("⚠️ Unstable Hemodynamic vs Clinical Outcomes")
        
        # Clean vital signs data
        df['SBP_clean'] = df['SBP'].str.extract(r'(\d+)').astype(float)
        df['DBP_clean'] = df['DBP'].str.extract(r'(\d+)').astype(float)
        df['SPO2_clean'] = df['SPO2%'].str.extract(r'(\d+)').astype(float)
        df['CBG_clean'] = df['CBG'].str.extract(r'(\d+)').astype(float)
        df['HR_clean'] = df['HR'].str.extract(r'(\d+)').astype(float)
        
        # Filter data for hemodynamic analysis
        filtered_df = df[['SBP_clean', 'DBP_clean', 'SPO2_clean', 'CBG_clean', 'HR_clean', 'CLINICAL OUTCOMES']].dropna()
        
        # Define unstable hemodynamics criteria
        filtered_df['unstable_hemo'] = (
            (filtered_df['SBP_clean'] < 120) |
            (filtered_df['DBP_clean'] < 80) |
            (filtered_df['SPO2_clean'] < 90) |
            (filtered_df['CBG_clean'] < 75) |
            (filtered_df['HR_clean'] < 45)
        )
        
        unstable_group = filtered_df[filtered_df['unstable_hemo'] == True]['CLINICAL OUTCOMES']
        stable_group = filtered_df[filtered_df['unstable_hemo'] == False]['CLINICAL OUTCOMES']
        
        # Create contingency table
        unstable_alive = len(unstable_group[unstable_group.str.upper() == 'ALIVE'])
        unstable_dead = len(unstable_group[unstable_group.str.upper() == 'DEAD'])
        stable_alive = len(stable_group[stable_group.str.upper() == 'ALIVE'])
        stable_dead = len(stable_group[stable_group.str.upper() == 'DEAD'])
        
        contingency_table = [[unstable_alive, unstable_dead], [stable_alive, stable_dead]]
        
        # Use Fisher's exact test if any cell has count < 5, otherwise chi-square
        if min(unstable_alive, unstable_dead, stable_alive, stable_dead) < 5:
            from scipy.stats import fisher_exact
            odds_ratio, p_chi2 = fisher_exact(contingency_table)
            chi2_stat = odds_ratio
            test_name = "Fisher's Exact Test"
        else:
            chi2_stat, p_chi2, dof, expected = chi2_contingency(contingency_table)
            test_name = "Chi-square Test"
        
        # Display test results
        st.subheader("📊 Statistical Test Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{test_name} Statistic", f"{chi2_stat:.4f}")
        with col2:
            st.metric("P-Value", f"{p_chi2:.4f}")
        with col3:
            significance = "Significant" if p_chi2 < 0.05 else "Not Significant"
            st.metric("Result", significance)
        
        # Criteria display
        st.subheader("🩺 Unstable Hemodynamic Criteria")
        st.write("**Patients classified as unstable if ANY of the following:**")
        st.write("• SBP < 120 mmHg")
        st.write("• DBP < 80 mmHg")
        st.write("• SPO2 < 90%")
        st.write("• CBG < 75 mg/dL")
        st.write("• HR < 45 BPM")
        
        # Contingency table display
        st.subheader("📋 Contingency Table")
        contingency_df = pd.DataFrame({
            'Unstable Hemodynamics': [unstable_alive, unstable_dead],
            'Stable Hemodynamics': [stable_alive, stable_dead]
        }, index=['ALIVE', 'DEAD'])
        st.dataframe(contingency_df, use_container_width=True)
        
        # Stacked bar chart
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name='ALIVE', x=['Unstable', 'Stable'], y=[unstable_alive, stable_alive], marker_color='#2E8B57'))
        fig_bar.add_trace(go.Bar(name='DEAD', x=['Unstable', 'Stable'], y=[unstable_dead, stable_dead], marker_color='#DC143C'))
        fig_bar.update_layout(
            title="Clinical Outcomes by Hemodynamic Status",
            xaxis_title="Hemodynamic Status",
            yaxis_title="Count",
            barmode='stack'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Survival rates
        st.subheader("📈 Survival Rates")
        col1, col2 = st.columns(2)
        
        with col1:
            unstable_total = unstable_alive + unstable_dead
            unstable_survival_rate = (unstable_alive / unstable_total * 100) if unstable_total > 0 else 0
            st.write("**Unstable Hemodynamics**")
            st.write(f"Total: {unstable_total}")
            st.write(f"Alive: {unstable_alive}")
            st.write(f"Dead: {unstable_dead}")
            st.write(f"Survival Rate: {unstable_survival_rate:.1f}%")
        
        with col2:
            stable_total = stable_alive + stable_dead
            stable_survival_rate = (stable_alive / stable_total * 100) if stable_total > 0 else 0
            st.write("**Stable Hemodynamics**")
            st.write(f"Total: {stable_total}")
            st.write(f"Alive: {stable_alive}")
            st.write(f"Dead: {stable_dead}")
            st.write(f"Survival Rate: {stable_survival_rate:.1f}%")
    
    elif analysis_type == "Combined Analysis":
        st.header("🔬 Combined Analysis Dashboard")
        
        # Prepare data for all tests
        initial_df = df[['INITIAL LACTATE (clean)', 'CLINICAL OUTCOMES']].dropna()
        clearance_df = df[['LACTATE CLEARANCE (clean)', 'CLINICAL OUTCOMES']].dropna()
        repeat_df = df[['REPEAT LACTATE (clean)', 'CLINICAL OUTCOMES']].dropna()
        age_df = df[['AGE', 'CLINICAL OUTCOMES']].dropna()
        cad_df = df[['K/C/O', 'CLINICAL OUTCOMES']].dropna()
        shtn_t2dm_df = df[['K/C/O', 'CLINICAL OUTCOMES']].dropna()
        
        # Clean vital signs for hemodynamic analysis
        df['SBP_clean'] = df['SBP'].str.extract(r'(\d+)').astype(float)
        df['DBP_clean'] = df['DBP'].str.extract(r'(\d+)').astype(float)
        df['SPO2_clean'] = df['SPO2%'].str.extract(r'(\d+)').astype(float)
        df['CBG_clean'] = df['CBG'].str.extract(r'(\d+)').astype(float)
        df['HR_clean'] = df['HR'].str.extract(r'(\d+)').astype(float)
        hemo_df = df[['SBP_clean', 'DBP_clean', 'SPO2_clean', 'CBG_clean', 'HR_clean', 'CLINICAL OUTCOMES']].dropna()
        
        # Initial Lactate groups
        initial_alive = initial_df[initial_df['CLINICAL OUTCOMES'] == 'ALIVE']['INITIAL LACTATE (clean)']
        initial_dead = initial_df[initial_df['CLINICAL OUTCOMES'] == 'DEAD']['INITIAL LACTATE (clean)']
        
        # Clearance groups
        clearance_alive = clearance_df[clearance_df['CLINICAL OUTCOMES'] == 'ALIVE']['LACTATE CLEARANCE (clean)']
        clearance_dead = clearance_df[clearance_df['CLINICAL OUTCOMES'] == 'DEAD']['LACTATE CLEARANCE (clean)']
        
        # Repeat lactate groups
        repeat_alive = repeat_df[repeat_df['CLINICAL OUTCOMES'] == 'ALIVE']['REPEAT LACTATE (clean)']
        repeat_dead = repeat_df[repeat_df['CLINICAL OUTCOMES'] == 'DEAD']['REPEAT LACTATE (clean)']
        
        # Age groups
        age_alive = age_df[age_df['CLINICAL OUTCOMES'] == 'ALIVE']['AGE']
        age_dead = age_df[age_df['CLINICAL OUTCOMES'] == 'DEAD']['AGE']
        
        # Perform tests
        u_stat_initial, p_val_initial = mannwhitneyu(initial_alive, initial_dead, alternative='two-sided')
        u_stat_clearance, p_val_clearance = mannwhitneyu(clearance_alive, clearance_dead, alternative='two-sided')
        u_stat_repeat, p_val_repeat = mannwhitneyu(repeat_alive, repeat_dead, alternative='two-sided')
        u_stat_age, p_val_age = mannwhitneyu(age_alive, age_dead, alternative='two-sided')
        
        # CAD analysis
        cad_df['has_CAD'] = cad_df['K/C/O'].str.contains('CAD', case=False, na=False)
        cad_group = cad_df[cad_df['has_CAD'] == True]['CLINICAL OUTCOMES']
        no_cad_group = cad_df[cad_df['has_CAD'] == False]['CLINICAL OUTCOMES']
        cad_alive_count = len(cad_group[cad_group.str.upper() == 'ALIVE'])
        cad_dead_count = len(cad_group[cad_group.str.upper() == 'DEAD'])
        no_cad_alive_count = len(no_cad_group[no_cad_group.str.upper() == 'ALIVE'])
        no_cad_dead_count = len(no_cad_group[no_cad_group.str.upper() == 'DEAD'])
        contingency = [[cad_alive_count, cad_dead_count], [no_cad_alive_count, no_cad_dead_count]]
        
        if min(cad_alive_count, cad_dead_count, no_cad_alive_count, no_cad_dead_count) < 5:
            from scipy.stats import fisher_exact
            chi2_stat_cad, p_val_cad = fisher_exact(contingency)
        else:
            chi2_stat_cad, p_val_cad, _, _ = chi2_contingency(contingency)
        
        # SHTN+T2DM analysis
        shtn_t2dm_df['has_SHTN_T2DM'] = (shtn_t2dm_df['K/C/O'].str.contains('SHTN', case=False, na=False) & 
                                         shtn_t2dm_df['K/C/O'].str.contains('T2DM', case=False, na=False))
        shtn_t2dm_group = shtn_t2dm_df[shtn_t2dm_df['has_SHTN_T2DM'] == True]['CLINICAL OUTCOMES']
        no_shtn_t2dm_group = shtn_t2dm_df[shtn_t2dm_df['has_SHTN_T2DM'] == False]['CLINICAL OUTCOMES']
        shtn_t2dm_alive_count = len(shtn_t2dm_group[shtn_t2dm_group.str.upper() == 'ALIVE'])
        shtn_t2dm_dead_count = len(shtn_t2dm_group[shtn_t2dm_group.str.upper() == 'DEAD'])
        no_shtn_t2dm_alive_count = len(no_shtn_t2dm_group[no_shtn_t2dm_group.str.upper() == 'ALIVE'])
        no_shtn_t2dm_dead_count = len(no_shtn_t2dm_group[no_shtn_t2dm_group.str.upper() == 'DEAD'])
        contingency_shtn_t2dm = [[shtn_t2dm_alive_count, shtn_t2dm_dead_count], [no_shtn_t2dm_alive_count, no_shtn_t2dm_dead_count]]
        
        if min(shtn_t2dm_alive_count, shtn_t2dm_dead_count, no_shtn_t2dm_alive_count, no_shtn_t2dm_dead_count) < 5:
            from scipy.stats import fisher_exact
            chi2_stat_shtn_t2dm, p_val_shtn_t2dm = fisher_exact(contingency_shtn_t2dm)
        else:
            chi2_stat_shtn_t2dm, p_val_shtn_t2dm, _, _ = chi2_contingency(contingency_shtn_t2dm)
        
        # Hemodynamic analysis
        hemo_df['unstable_hemo'] = (
            (hemo_df['SBP_clean'] < 120) |
            (hemo_df['DBP_clean'] < 80) |
            (hemo_df['SPO2_clean'] < 90) |
            (hemo_df['CBG_clean'] < 75) |
            (hemo_df['HR_clean'] < 45)
        )
        unstable_hemo_group = hemo_df[hemo_df['unstable_hemo'] == True]['CLINICAL OUTCOMES']
        stable_hemo_group = hemo_df[hemo_df['unstable_hemo'] == False]['CLINICAL OUTCOMES']
        unstable_hemo_alive = len(unstable_hemo_group[unstable_hemo_group.str.upper() == 'ALIVE'])
        unstable_hemo_dead = len(unstable_hemo_group[unstable_hemo_group.str.upper() == 'DEAD'])
        stable_hemo_alive = len(stable_hemo_group[stable_hemo_group.str.upper() == 'ALIVE'])
        stable_hemo_dead = len(stable_hemo_group[stable_hemo_group.str.upper() == 'DEAD'])
        contingency_hemo = [[unstable_hemo_alive, unstable_hemo_dead], [stable_hemo_alive, stable_hemo_dead]]
        
        if min(unstable_hemo_alive, unstable_hemo_dead, stable_hemo_alive, stable_hemo_dead) < 5:
            from scipy.stats import fisher_exact
            chi2_stat_hemo, p_val_hemo = fisher_exact(contingency_hemo)
        else:
            chi2_stat_hemo, p_val_hemo, _, _ = chi2_contingency(contingency_hemo)
        
        # Test results table
        st.subheader("🧪 Statistical Test Results")
        results_data = {
            'Test': ['Initial Lactate vs Outcomes', 'Lactate Clearance vs Outcomes', 'Repeat Lactate vs Outcomes', 'Age vs Outcomes', 'CAD vs Outcomes', 'SHTN+T2DM vs Outcomes', 'Unstable Hemodynamics vs Outcomes'],
            'Test Type': ['Mann-Whitney U', 'Mann-Whitney U', 'Mann-Whitney U', 'Mann-Whitney U', 'Fisher/Chi-square', 'Fisher/Chi-square', 'Fisher/Chi-square'],
            'Statistic': [u_stat_initial, u_stat_clearance, u_stat_repeat, u_stat_age, chi2_stat_cad, chi2_stat_shtn_t2dm, chi2_stat_hemo],
            'P-Value': [p_val_initial, p_val_clearance, p_val_repeat, p_val_age, p_val_cad, p_val_shtn_t2dm, p_val_hemo],
            'Significant (α=0.05)': [
                'Yes' if p_val_initial < 0.05 else 'No',
                'Yes' if p_val_clearance < 0.05 else 'No',
                'Yes' if p_val_repeat < 0.05 else 'No',
                'Yes' if p_val_age < 0.05 else 'No',
                'Yes' if p_val_cad < 0.05 else 'No',
                'Yes' if p_val_shtn_t2dm < 0.05 else 'No',
                'Yes' if p_val_hemo < 0.05 else 'No'
            ]
        }
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)
        
        # Side-by-side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            # Initial Lactate scatter plot
            fig_scatter1 = px.scatter(
                initial_df, 
                x='CLINICAL OUTCOMES', 
                y='INITIAL LACTATE (clean)',
                color='CLINICAL OUTCOMES',
                title="Initial Lactate by Outcome",
                color_discrete_map={'ALIVE': '#2E8B57', 'DEAD': '#DC143C'}
            )
            fig_scatter1.add_hline(y=initial_df['INITIAL LACTATE (clean)'].mean(), 
                                 line_dash="dash", line_color="gray",
                                 annotation_text=f"Mean: {initial_df['INITIAL LACTATE (clean)'].mean():.2f}")
            st.plotly_chart(fig_scatter1, use_container_width=True)
        
        with col2:
            # Lactate Clearance scatter plot
            fig_scatter2 = px.scatter(
                clearance_df, 
                x='CLINICAL OUTCOMES', 
                y='LACTATE CLEARANCE (clean)',
                color='CLINICAL OUTCOMES',
                title="Lactate Clearance by Outcome",
                color_discrete_map={'ALIVE': '#2E8B57', 'DEAD': '#DC143C'}
            )
            fig_scatter2.add_hline(y=clearance_df['LACTATE CLEARANCE (clean)'].mean(), 
                                 line_dash="dash", line_color="gray",
                                 annotation_text=f"Mean: {clearance_df['LACTATE CLEARANCE (clean)'].mean():.2f}")
            st.plotly_chart(fig_scatter2, use_container_width=True)
        
        # Correlation analysis
        st.subheader("🔗 Correlation Analysis")
        correlation_df = df[['INITIAL LACTATE (clean)', 'LACTATE CLEARANCE (clean)']].dropna()
        correlation = correlation_df.corr().iloc[0, 1]
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Correlation Coefficient", f"{correlation:.3f}")
        
        with col2:
            fig_corr = px.scatter(
                correlation_df,
                x='INITIAL LACTATE (clean)',
                y='LACTATE CLEARANCE (clean)',
                title="Initial Lactate vs Lactate Clearance"
            )
            st.plotly_chart(fig_corr, use_container_width=True)

except FileNotFoundError:
    st.error("❌ CSV file not found. Please upload your data file using the sidebar.")
    st.info("💡 Tip: Use the file uploader in the sidebar or generate sample data to test the dashboard.")
except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")
    st.info("💡 Please check your data format and try again.")

# Footer
st.markdown("---")
st.markdown("*Dashboard created for medical data analysis using multiple statistical tests*")