import json
import re
import argparse
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from collections import defaultdict

import matplotlib
print(matplotlib.style.available)
pd.set_option('max_colwidth', None)
pd.set_option('display.max_rows', None)


class DataFrame:
    def __init__(self, data: list[dict], use_gpt: bool = False):
        # Building a dataframe
        self.df = pd.DataFrame(data, columns=[
            "obs_semester",
            "prop_tags",
            "prop_comments",
            "coi_names",
            "prop_status",
            "prop_abstract",
            "pi_email",
            "obs_tel",
            "pi_affiliation",
            "pi_name",
            "prop_allocated",
            "prop_id",
            "prop_title",
            # "obs_year",
            # "prop_subject",
        ])
        self._preprocess()

        self.subjects = "planet, star, galaxy, star cluster, galaxy cluster, nebulae, novae, black hole"
        self.tag_subjects(use_gpt)


    def _preprocess(self):
        for i in range(len(self.df)):
            # Preprocess "obs_semester" to "obs_year"
            # For example, '2020A': str -> 2020: int
            self.df.loc[i, 'obs_year'] = int(self.df.loc[i, 'obs_semester'][:4])
            self.df['obs_year'] = self.df['obs_year'].fillna(-1)
            self.df['obs_year'] = self.df['obs_year'].astype(int)

            # Preprocessing "prop_allocated"
            # For example, "1 night": str -> 12.0: float
            if 'hour' not in self.df.loc[i, 'prop_allocated']:
                try:
                    match = re.search(r'([\d.]+)\snight', self.df.loc[i, 'prop_allocated'])
                    self.df.loc[i, 'prop_allocated'] = float(match.group(1)) * 12
                except AttributeError:
                    print('Cannot parse the string. No "night" found.')
            else:
                try:
                    match = re.search(r'([\d.]+)\shour', self.df.loc[i, 'prop_allocated'])
                    self.df.loc[i, 'prop_allocated'] = float(match.group(1))
                except AttributeError:
                    print('Cannot parse the string. No "hour" found.')

    def tag_subjects(self, use_gpt: bool):
        if use_gpt:
            subjects = []
            client = OpenAI(
                api_key="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"  # Insert your API key
            )
            for title in tqdm(self.df['prop_title'], desc='Retrieving responses from GPT'):
                sys_message = f"""
                    Your job is to distinguish which subject the given text is related to.
                    Only provide the subject, without explaining the reason you chose.
                    If the text is highly related to two or more subjects, then display all the subjects.
                    You must choose the subject only in [{self.subjects}, others].
                    If there are two or more subjects, they must be separated by comma.
                    Do not include any symbols or punctuation other than commas in your response.
                    Say 'others', only if the title cannot be specified into any given subjects.
                """
                completion = client.chat.completions.create(
                    model='gpt-3.5-turbo',
                    messages=[{'role': 'system', 'content': sys_message}, {'role': 'user', 'content': f'Text: {title}'}],
                    top_p=0
                )
                result = completion.choices[0].message.content
                subjects.append([sbj.strip() for sbj in result.split(',')])
            self.df['prop_subject'] = subjects
        else:
            try:
                self.df['prop_subject'] = [['star'], ['nebulae'], ['galaxy', 'star cluster'], ['galaxy', 'others'], ['galaxy cluster'], ['nebulae'], ['black hole'], ['galaxy cluster'], ['galaxy cluster'], ['galaxy', 'others'], ['galaxy'], ['galaxy', 'others'], ['galaxy'], ['galaxy', 'others'], ['galaxy', 'star cluster'], ['galaxy cluster', 'others'], ['galaxy', 'galaxy cluster'], ['galaxy cluster'], ['galaxy'], ['galaxy cluster'], ['galaxy', 'others'], ['galaxy', 'others'], ['galaxy', 'others'], ['galaxy'], ['galaxy'], ['star'], ['black hole'], ['others'], ['galaxy cluster'], ['galaxy', 'others'], ['galaxy'], ['galaxy'], ['novae'], ['star'], ['galaxy', 'others'], ['planet'], ['galaxy', 'star'], ['galaxy'], ['galaxy'], ['galaxy', 'others'], ['planet', 'nebulae'], ['galaxy', 'others'], ['galaxy', 'others'], ['galaxy', 'others'], ['galaxy cluster'], ['galaxy', 'others'], ['black hole'], ['star', 'others'], ['galaxy', 'others'], ['galaxy'], ['galaxy', 'others'], ['galaxy', 'others'], ['galaxy cluster'], ['novae'], ['others'], ['galaxy'], ['galaxy', 'galaxy cluster'], ['galaxy', 'others'], ['star'], ['planet'], ['star cluster'], ['galaxy'], ['black hole'], ['galaxy', 'star cluster'], ['galaxy'], ['galaxy'], ['galaxy', 'others'], ['others'], ['galaxy', 'others'], ['star cluster'], ['star'], ['galaxy', 'others'], ['novae'], ['galaxy cluster'], ['star cluster', 'galaxy'], ['star', 'star cluster'], ['galaxy'], ['galaxy'], ['galaxy cluster'], ['nebulae', 'planet'], ['galaxy'], ['star cluster', 'others'], ['planet'], ['star'], ['galaxy', 'nebulae'], ['galaxy', 'others'], ['galaxy'], ['galaxy', 'others'], ['galaxy'], ['star'], ['novae'], ['planet'], ['others'], ['star'], ['star'], ['galaxy'], ['galaxy'], ['star'], ['planet'], ['star cluster', 'galaxy'], ['galaxy cluster'], ['star cluster'], ['novae'], ['galaxy'], ['galaxy', 'galaxy cluster'], ['galaxy', 'star'], ['star cluster'], ['galaxy'], ['others'], ['galaxy', 'others'], ['star'], ['galaxy', 'nebulae'], ['others'], ['galaxy', 'others'], ['galaxy'], ['others'], ['star', 'others'], ['star'], ['star'], ['galaxy', 'star cluster'], ['galaxy', 'others'], ['star', 'others'], ['novae'], ['star'], ['galaxy', 'others'], ['star', 'others'], ['star cluster', 'galaxy'], ['planet'], ['galaxy', 'star cluster'], ['galaxy cluster'], ['star', 'others'], ['star'], ['others'], ['star'], ['star', 'others'], ['galaxy'], ['galaxy'], ['galaxy', 'others'], ['novae'], ['galaxy', 'star'], ['star', 'others'], ['galaxy cluster', 'others'], ['star', 'planet'], ['galaxy'], ['star'], ['others'], ['others'], ['galaxy', 'others'], ['galaxy cluster'], ['star'], ['galaxy'], ['novae'], ['galaxy', 'others'], ['galaxy'], ['galaxy', 'star cluster'], ['galaxy', 'others'], ['planet'], ['others'], ['others'], ['star', 'others'], ['star'], ['star'], ['galaxy', 'star'], ['galaxy'], ['galaxy', 'others'], ['galaxy'], ['galaxy'], ['galaxy cluster'], ['others'], ['black hole'], ['galaxy'], ['galaxy'], ['galaxy'], ['galaxy'], ['star'], ['galaxy'], ['galaxy'], ['galaxy'], ['black hole', 'galaxy'], ['galaxy cluster'], ['star'], ['nebulae'], ['galaxy'], ['others'], ['galaxy'], ['galaxy'], ['galaxy', 'others'], ['galaxy cluster'], ['planet'], ['black hole'], ['galaxy cluster'], ['galaxy', 'others'], ['star'], ['novae'], ['galaxy', 'others'], ['galaxy', 'others'], ['galaxy', 'others'], ['star'], ['galaxy', 'others'], ['galaxy'], ['galaxy', 'others'], ['galaxy'], ['galaxy'], ['galaxy', 'others'], ['galaxy'], ['galaxy cluster'], ['novae'], ['black hole'], ['galaxy cluster'], ['others'], ['star'], ['others'], ['novae'], ['star'], ['galaxy', 'star'], ['galaxy', 'others'], ['star'], ['others'], ['galaxy'], ['galaxy', 'others'], ['galaxy cluster'], ['galaxy'], ['galaxy'], ['galaxy'], ['galaxy'], ['black hole'], ['others'], ['star', 'others'], ['star', 'others'], ['galaxy', 'nebulae'], ['galaxy', 'others'], ['star cluster'], ['star cluster'], ['galaxy cluster'], ['star cluster'], ['star cluster'], ['nebulae'], ['galaxy', 'star'], ['nebulae'], ['star', 'nebulae'], ['nebulae'], ['galaxy'], ['galaxy'], ['star cluster'], ['black hole'], ['nebulae'], ['nebulae'], ['galaxy cluster'], ['galaxy cluster', 'galaxy'], ['galaxy', 'star cluster'], ['black hole'], ['galaxy cluster'], ['nebulae'], ['galaxy'], ['galaxy cluster'], ['star', 'nebulae'], ['galaxy'], ['nebulae'], ['nebulae'], ['galaxy'], ['galaxy cluster'], ['star', 'nebulae'], ['galaxy'], ['nebulae'], ['star'], ['star'], ['star'], ['galaxy'], ['galaxy'], ['galaxy'], ['galaxy'], ['galaxy'], ['galaxy cluster'], ['galaxy'], ['galaxy cluster'], ['galaxy cluster'], ['star cluster'], ['galaxy'], ['galaxy cluster', 'black hole'], ['galaxy', 'others']]
            except:
                print('This error should occur when the website page is updated. '
                      'You should turn on the "use_gpt" option to True.')

    def plot_prop_counts(self):
        """Plot the number of proposals in each semester."""
        unique, counts = np.unique(self.df['obs_semester'], return_counts=True)
        unique_obs_semester = dict(zip(unique, counts))

        plt.bar(range(len(unique_obs_semester)), unique_obs_semester.values(), align='center')
        plt.xticks(range(len(unique_obs_semester)), unique_obs_semester.keys(), rotation=45)
        plt.yticks(np.arange(0.0, 27.0, 3.0))
        plt.xlabel('# of proposals')
        plt.ylabel('Semester')

    def plot_years_hours(self):
        """Plot the time observed in each semester."""
        years = np.unique(self.df['obs_year'])
        allocated_hours = [self.df[self.df['obs_year'] == year]['prop_allocated'].sum() for year in years]

        plt.bar(years, allocated_hours)
        plt.xticks(range(years[0], years[-1]+1))
        plt.xlabel('Years')
        plt.ylabel('Hours')

    def plot_subject_hours(self):
        """Plot how many times a subject takes when observing."""
        subjects = [subj.strip() for subj in self.subjects.split(',')]
        recent_df = self.df[self.df['obs_year'] >= 2020]
        allocated_hours = [recent_df[recent_df['prop_subject'].apply(lambda x: subject in x)]
                           ['prop_allocated'].sum() for subject in subjects]

        plt.bar(subjects, allocated_hours)
        plt.xticks(rotation=45)
        plt.xlabel(f'Subjects (year ≥ 2020)')
        plt.ylabel('Hours')

    def plot_subject_device(self):
        """Plot what device are frequently used for a subject."""
        subjects = [subj.strip() for subj in self.subjects.split(',')]
        devices = pd.unique(self.df['obs_tel'])
        device_counts = defaultdict()
        for subject in subjects:
            device_counts[subject] = \
                dict(self.df[self.df['prop_subject'].apply(lambda x: subject in x)]['obs_tel'].value_counts())
        for key in device_counts:
            for device in devices:
                if device not in device_counts[key]:
                    device_counts[key][device] = 0
        device_counts = pd.DataFrame(device_counts)

        ax1 = plt.subplot2grid(shape=(1, 3), loc=(0, 0), colspan=1)
        device_counts.plot.bar(ax=ax1, stacked=True)
        ax1.set_ylabel('# of props')

        ax2 = plt.subplot2grid(shape=(1, 3), loc=(0, 1), colspan=2)
        device_counts.T.plot.bar(ax=ax2)
        ax2.set_ylabel('% of props')


def main(data_path: str, style: str, use_gpt: bool) -> None:
    try:
        resp = requests.get("http://accepted.kgmt-sp.appspot.com/db/accepted_proposals_phase2_jsonp")
        html = resp.text
        raw_data = json.loads(html[1:-1])   # parsing part
        data = raw_data['data']             # parsing part
    except:
        print('This error should occur when: \n'
              '\t1. The cite has been moved. -> Check if the link is still valid.'
              '\t2. The format of the cite content may have changed. -> Check the parsing part.')

    df = DataFrame(data, use_gpt)

    plt.style.use(style)
    fig1 = plt.figure()
    df.plot_prop_counts()
    fig1.tight_layout()

    fig2 = plt.figure()
    df.plot_years_hours()
    fig2.tight_layout()

    fig3 = plt.figure()
    df.plot_subject_hours()
    fig3.tight_layout()

    fig4 = plt.figure()
    df.plot_subject_device()
    fig4.tight_layout()

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default='../data/K-GMT.json')
    parser.add_argument("--pyplot_style", default='bmh')
    parser.add_argument("--use_gpt", default=False)
    args = parser.parse_args()

    main(data_path=args.data_path, style=args.pyplot_style, use_gpt=args.use_gpt)
