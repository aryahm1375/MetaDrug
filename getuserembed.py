import numpy as np
from openai import OpenAI

api_key = "sk-proj-x3954WNtiMxjmkMn3MqUT3BlbkFJBClEYjfsHOBmWM5cB2Vx"
import pyhealth
from pyhealth.datasets import get_dataloader#, split_by_visit
from Uitls.CustomSplitter import split_by_patient
from Uitls import CustomMIMIC
import Uitls.MimicMultiHot
from MELU.AKIDataset import CustomMIMIC as AKIS
from pyhealth.medcode import InnerMap
g_timestep = 5

client = OpenAI(api_key=api_key)
def get_LLM_code(code, title, model="text-embedding-3-small"):
    code_type = 'ATC Level 3'
    lookup = InnerMap.load("ATC")

    if title == 'diagnosis':
        code_type = 'ICD9'
        lookup = InnerMap.load("ICD9CM")
    if title == 'procedure':
        code_type = 'ICD9'
        lookup = InnerMap.load("ICD9PROC")
    print('code', code, title)
    text = f'A {title} code for Electronic Health Records in {code_type} with the following name and description:\n' \
           f'code: {code}, code name: {lookup.lookup(code)}'
    print('TEXT:', text)
    return client.embeddings.create(input=[text], model=model, dimensions=256).data[0].embedding


def get_embedding(text, model="text-embedding-3-small", dimensions=None):
    input = 'The folliwing medical visit includes medication, procedure, and diagnosis codes as follows:\n'
    input = input + text
    print(input)
    return client.embeddings.create(input=[input], model=model, dimensions=dimensions).data[0].embedding

tokens = ['A01A', 'A02A', 'A02B', 'A03A', 'A03B', 'A03C', 'A03F', 'A04A', 'A05A', 'A06A', 'A07A', 'A07B', 'A07D', 'A07E', 'A07F', 'A07X', 'A08A', 'A09A', 'A10A', 'A10B', 'A11C', 'A11D', 'A11G', 'A11H', 'A12A', 'A12B', 'A12C', 'A14A', 'A16A', 'B01A', 'B02A', 'B02B', 'B03A', 'B03B', 'B03X', 'B05A', 'B05B', 'B05C', 'B05X', 'B06A', 'C01A', 'C01B', 'C01C', 'C01D', 'C01E', 'C02A', 'C02C', 'C02D', 'C02L', 'C03A', 'C03B', 'C03C', 'C03D', 'C03E', 'C03X', 'C04A', 'C05A', 'C05B', 'C05C', 'C07A', 'C08C', 'C08D', 'C09A', 'C09C', 'C09X', 'C10A', 'D01A', 'D01B', 'D02A', 'D03A', 'D04A', 'D05A', 'D05B', 'D06A', 'D06B', 'D07A', 'D07C', 'D07X', 'D08A', 'D09A', 'D10A', 'D11A', 'G01A', 'G03A', 'G03B', 'G03C', 'G03D', 'G03F', 'G03G', 'G03X', 'G04B', 'G04C', 'H01A', 'H01B', 'H01C', 'H02A', 'H02C', 'H03A', 'H03B', 'H04A', 'H05B', 'J01A', 'J01C', 'J01D', 'J01E', 'J01F', 'J01G', 'J01M', 'J01X', 'J02A', 'J04A', 'J04B', 'J05A', 'J06B', 'J07A', 'J07B', 'J07C', 'L01A', 'L01B', 'L01C', 'L01D', 'L01E', 'L01X', 'L02A', 'L02B', 'L03A', 'L04A', 'M01A', 'M01C', 'M02A', 'M03A', 'M03B', 'M03C', 'M04A', 'M05B', 'N01A', 'N01B', 'N02A', 'N02B', 'N02C', 'N03A', 'N04A', 'N04B', 'N05A', 'N05B', 'N05C', 'N06A', 'N06B', 'N06D', 'N07A', 'N07B', 'N07X', 'P01A', 'P01B', 'P01C', 'P02C', 'R01A', 'R01B', 'R02A', 'R03A', 'R03B', 'R03C', 'R03D', 'R05C', 'R05D', 'R05F', 'R06A', 'R07A', 'S01A', 'S01B', 'S01C', 'S01E', 'S01F', 'S01G', 'S01H', 'S01X', 'S02A', 'S03A', 'S03B', 'S03C', 'V03A', 'V04C', 'V06D']


def save_llm_embed(samples, patient):
    diagnosis_look = InnerMap.load("ICD9CM")
    procedure_look = InnerMap.load("ICD9PROC")
    atc_look = InnerMap.load("ATC")
    if len(samples) > 1:
        input_samples = samples[-1]
        all_conditions = input_samples['conditions']
        all_procedures = input_samples['procedures']
        all_drug_history = input_samples['drugs_hist']
        visits_embed = []
        for vst in range(len(all_conditions)):
            text = ''
            drugs_atc = ''
            procedures_icd = ''
            conditions_icd = ''
            if len(all_conditions[vst]) > 0:
                for item in all_conditions[vst]:
                    try:
                        conditions_icd = conditions_icd + diagnosis_look.lookup(item) + ' , '
                    except:
                        print('error')
            if len(all_procedures[vst]) > 0:
                for item in all_procedures[vst]:
                    try:
                        procedures_icd = procedures_icd + procedure_look.lookup(item) + ' , '
                    except:
                        print('error')
            if len(all_drug_history[vst]) > 0:
                for item in all_drug_history[vst]:
                    try:
                        drugs_atc = drugs_atc + atc_look.lookup(item) + ' , '
                    except:
                        print('error')
            if not conditions_icd == '':
                text = text + "diagnosis codes: " + conditions_icd + '\n'
            if not procedures_icd == '':
                text = text + "procedure codes: " + procedures_icd + '\n'

            visit_embedding = get_embedding(text, model='text-embedding-3-small', dimensions=1024)
            visits_embed.append(visit_embedding)

        np.save(f'UserEmbeddings/{patient.patient_id}', np.array(visits_embed))

llm_code_embeddings = {}

def encode_llm(input_list, dict_input, title = 'medication'):
    for input in input_list:
        if not input in dict_input.keys():
            embedding = get_LLM_code(input, title)
            dict_input[input] = embedding
    return dict_input


def sequential_drugs_attribute(patient, retrieval = False):
    samples = []
    global llm_code_embeddings
    sequential_conditions = []
    sequential_procedures = []
    sequential_drugs = []   # not include the drugs now
    for visit in patient:
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        drugs = [drug for drug in drugs if drug in tokens]
        try:
            llm_code_embeddings = encode_llm(procedures, llm_code_embeddings, 'procedure')
            llm_code_embeddings = encode_llm(conditions, llm_code_embeddings, 'diagnosis')
            llm_code_embeddings = encode_llm(drugs, llm_code_embeddings, 'medication')
        except:
            print('Error')

        sequential_conditions.append(conditions)
        sequential_procedures.append(procedures)
        if len(drugs) == 0:
            sequential_drugs.append([])
            continue
        sequential_drugs.append([])
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": sequential_conditions.copy(),
                "procedures": sequential_procedures.copy(),
                "drugs_hist": sequential_drugs.copy(),
                "drugs": drugs,
            }
        )
        sequential_drugs[-1] = drugs

    return samples
av_num_visits = []
def sequential_drug_recommendation_retain(patient, retrieval = False):
    samples = []
    sequential_conditions = []
    sequential_procedures = []
    sequential_drugs = [] # not include the drugs now
    global av_num_visits

    n_visits_cound = len(patient)
    for visit in patient:
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        drugs = [drug for drug in drugs if drug in tokens]

        sequential_conditions.append(conditions)
        sequential_procedures.append(procedures)
        if len(drugs) == 0:
            sequential_drugs.append([])
            continue
        sequential_drugs.append([])
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": sequential_conditions.copy(),
                "procedures": sequential_procedures.copy(),
                "drugs_hist": sequential_drugs.copy(),
                "drugs": drugs,
            }
        )
        sequential_drugs[-1] = drugs
    if len(samples) > 1:
        av_num_visits.append(len(samples))
        return [samples[-1]]
    else:
        return []

def sequential_drug_recommendation(patient):
    samples = []

    sequential_conditions = []
    sequential_procedures = []
    sequential_drugs = [] # not include the drugs now
    for visit in patient:

        # step 1: obtain feature information
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        drugs = [drug for drug in drugs if not drug in ['V09I', 'A05B', 'D03B', 'P02B']]
        if len(drugs) == 0:
            continue
        sequential_conditions.append(conditions)
        sequential_procedures.append(procedures)
        sequential_drugs.append([])

        if len(drugs) == 0:
            sequential_drugs[-1] = drugs
            continue

        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": sequential_drugs.copy()[-g_timestep:],
                "procedures": sequential_procedures.copy()[-g_timestep:],
                "drugs_hist": sequential_drugs.copy()[-g_timestep:],
                "drugs": drugs,
            }
        )
        sequential_drugs[-1] = drugs







    return samples


def get_loader(value = None, cross = 1, timestep = 5):
    global g_timestep
    dataset = CustomMIMIC.MIMIC3Dataset(
            root="/home/a053h213/PycharmProjects/PrescriptionRecommendation/Datasets/MIMICIII/data",
            tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
        dev=True,
        #refresh_cache=True

    )
    g_timestep = timestep
    dataset = dataset.set_task(task_fn=sequential_drug_recommendation)
    print(len(dataset))
    train_ds, val_ds, test_ds = split_by_patient(dataset, [0.8, 0.0, 0.2], seed=cross)
    val_ds = test_ds
    train_loader = get_dataloader(train_ds, batch_size=128, shuffle=False)
    val_loader = get_dataloader(val_ds, batch_size=128, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=128, shuffle=False)
    return dataset, train_loader, test_loader, test_loader
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
def get_llm_embed():
    user = {}
    for root, directories, filenames in os.walk('UserEmbeddings/'):
        for filename in filenames:
            patient_visits = np.load(f'UserEmbeddings/{filename}')
            user[filename.replace('.npy','')] = np.sum(patient_visits, axis = 0)
            print(user[filename.replace('.npy','')])
    with open('user_llm_dict.pkl', 'wb') as file:
        pickle.dump(user, file)
            #print(np.mean(patient_visits, axis = 0))
def get_similarity_embed(input_ids, train_ids, label = 'train', cross = 0):
    user_similarity = {}
    with open('user_llm_dict.pkl', 'rb') as file:
        loaded_llm = pickle.load(file)
    for input in input_ids:
        similar_user = []
        main_user_embed = loaded_llm[input]
        for tr in train_ids:
            if not input == tr:
                us = loaded_llm[tr]
                similarity = cosine_similarity([main_user_embed, us])[0,1] #np.dot(main_user_embed, us)/(np.linalg.norm(main_user_embed)*np.linalg.norm(us))
                #print(similarity)
                similar_user.append([tr, similarity])
                #print(similar_user)
        similar_sorted = sorted(similar_user, key=lambda x: x[1], reverse=True)[0:10]
        most_similar_patients = [similar_sorted[i][0] for i in range(len(similar_sorted))]
        user_similarity[input] = most_similar_patients
    with open(f'user_similarity_{label}_{cross}.pkl', 'wb') as file:
        pickle.dump(user_similarity, file)
    print(user_similarity)

def get_similar_patients(value = None, cross = 0, timestep = 5):
    dataset = CustomMIMIC.MIMIC3Dataset(
        root="/home/a053h213/PycharmProjects/PrescriptionRecommendation/Datasets/MIMICIII/data",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},

    )

def get_patient_LLM(cross = 0):
    with open(f'/home/a053h213/PycharmProjects/pythonProject/LLM/user_similarity_train_{cross}.pkl', 'rb') as file:
        train_llm = pickle.load(file)
    with open(f'/home/a053h213/PycharmProjects/pythonProject/LLM/user_similarity_test_{cross}.pkl', 'rb') as file:
        test_llm = pickle.load(file)
    train_llm.update(test_llm)
    return train_llm

def retrieve_indexes(cross):
    dataset = CustomMIMIC.MIMIC3Dataset(
        root="/home/a053h213/PycharmProjects/PrescriptionRecommendation/Datasets/MIMICIII/data",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
    )
    dataset, _ = dataset.set_task(task_fn=sequential_drug_recommendation_retain, retrieval=False,
                                  patient_visits=None, patient_LLM=None)

    train_ds, val_ds, test_ds, train_patient_ids, val_patient_ids, test_patient_ids = split_by_patient(dataset,
                                                                                                       [0.8, 0.0, 0.2],
                                                                                                       seed=cross)
    return train_patient_ids, test_patient_ids


def get_loader_retain(value = None, cross = 0, timestep = 5):
    global g_timestep
    train_file_name = f'/home/a053h213/PycharmProjects/pythonProject/LLM/ProcessedData/train_{cross}_jaccard.pkl'
    test_file_name = f'/home/a053h213/PycharmProjects/pythonProject/LLM/ProcessedData/test_{cross}_jaccard.pkl'
    dataset_file_name = f'/home/a053h213/PycharmProjects/pythonProject/LLM/ProcessedData/dataset_{cross}_jaccard.pkl'
    print(os.path.isfile(test_file_name), os.path.isfile(train_file_name))
    if (not os.path.isfile(train_file_name)) or (not os.path.isfile(test_file_name)):
        dataset = CustomMIMIC.MIMIC3Dataset(
                root="/home/a053h213/PycharmProjects/PrescriptionRecommendation/Datasets/MIMICIII/data",
                tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
            code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},

        )
        g_timestep = timestep

        similar_patients = get_patient_LLM(cross)
        _, patient_dict = dataset.set_task(task_fn=sequential_drug_recommendation_retain, retrieval=True)
        dataset, _ = dataset.set_task(task_fn=sequential_drug_recommendation_retain, retrieval=False, patient_visits=patient_dict, patient_LLM=similar_patients, retrieve_ids=True)
        train_ds, val_ds, test_ds, train_patient_ids, val_patient_ids, test_patient_ids = split_by_patient(dataset, [0.8, 0.0, 0.2], seed=cross)


        val_ds = test_ds
        train_loader = get_dataloader(train_ds, batch_size=128, shuffle=False)
        val_loader = get_dataloader(val_ds, batch_size=128, shuffle=False)
        test_loader = get_dataloader(test_ds, batch_size=128, shuffle=False)
        with open(train_file_name, 'wb') as file:
            pickle.dump(train_loader, file)
        with open(test_file_name, 'wb') as file:
            pickle.dump(test_loader, file)
        with open(dataset_file_name, 'wb') as file:
            pickle.dump(dataset, file)
    else:
        with open(train_file_name, 'rb') as file:
            train_loader = pickle.load(file)
        with open(test_file_name, 'rb') as file:
            test_loader = pickle.load(file)
        with open(dataset_file_name, 'rb') as file:
            dataset = pickle.load(file)

    return dataset, train_loader, test_loader, test_loader

def get_loader_all_codes(value = None, cross = 0, timestep = 5):
    #train_file_name = f'/home/a053h213/PycharmProjects/pythonProject/LLM/ProcessedData/train_{cross}.pkl'
    #test_file_name = f'/home/a053h213/PycharmProjects/pythonProject/LLM/ProcessedData/test_{cross}.pkl'
    #dataset_file_name = f'/home/a053h213/PycharmProjects/pythonProject/LLM/ProcessedData/dataset_{cross}.pkl'
    dataset = Uitls.CustomMIMIC.MIMIC3Dataset(
        root="/home/a053h213/PycharmProjects/PrescriptionRecommendation/Datasets/MIMICIII/data",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},

    )
    g_timestep = timestep
    similar_patients = get_patient_LLM(cross)
    _, patient_dict = dataset.set_task(task_fn=sequential_drug_recommendation_retain, retrieval=True)

def get_loader_retain_status(value = None, cross = 0, timestep = 5):
    global g_timestep
    train_file_name = f'/home/a053h213/PycharmProjects/pythonProject/LLM/ProcessedData/train_{cross}_jaccard.pkl'
    test_file_name = f'/home/a053h213/PycharmProjects/pythonProject/LLM/ProcessedData/test_{cross}_jaccard.pkl'
    dataset_file_name = f'/home/a053h213/PycharmProjects/pythonProject/LLM/ProcessedData/dataset_{cross}_jaccard.pkl'
    dataset = pyhealth.datasets.MIMIC3Dataset(
            root="/home/a053h213/PycharmProjects/PrescriptionRecommendation/Datasets/MIMICIII/data",
            tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},

    )
    g_timestep = timestep

    dataset = dataset.set_task(task_fn=sequential_drug_recommendation_retain)
    print(av_num_visits)
    print('av_visits',np.mean(np.array(av_num_visits)))

    dataset.stat()
    dataset.info()
    return None


def sequential_drug_recommendation_AKI(patient):
    samples = []
    #print('******************')
    global DX
    global RX
    global PX
    global av_num_visits

    sequential_conditions = []
    sequential_procedures = []
    sequential_drugs = []
    for visit in patient:
        conditions = visit.get_code_list(table="AKI_DX_CURRENT")
        procedures = visit.get_code_list(table="AKI_PX")
        drugs = visit.get_code_list(table="AKI_PMED")

        sequential_conditions.append(conditions)
        sequential_procedures.append(procedures)
        if len(drugs) == 0:
            sequential_drugs.append([])
            continue
        sequential_drugs.append([])
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": sequential_conditions.copy(),
                "procedures": sequential_procedures.copy(),
                "drugs_hist": sequential_drugs.copy(),
                "drugs": drugs,
            }
        )
        sequential_drugs[-1] = drugs
    if len(samples) > 1:
        av_num_visits.append(len(samples))
        return [samples[-1]]
    else:
        return []
def get_loader_retain_status_AKI(value = None, cross = 0, timestep = 5):
    global g_timestep
    print('Procesing_AKI')
    dataset = AKIS(
            root="/home/a053h213/graph/DeepAKI/raw",
            tables=["AKI_DX_CURRENT", "AKI_PX", "AKI_PMED"],
        code_mapping={"RxNorm": ("ATC", {"target_kwargs": {"level": 3}})},

    )
    g_timestep = timestep

    dataset = dataset.set_task(task_fn=sequential_drug_recommendation_AKI)
    print(av_num_visits)
    print('av_visits', np.mean(np.array(av_num_visits)))
    dataset.stat()
    dataset.info()
    return None

get_loader_retain_status_AKI()