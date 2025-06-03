import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Process some integers.")

    # Add arguments
    parser.add_argument('--seed', type=int, help='seed value for folding', default=0)
    parser.add_argument('--dim', type=int, help='dimention of the NN',default=256)
    parser.add_argument('--title', type=str, help='titles',default='check')
    parser.add_argument('--user', action='store_true', help='User Adaptation')
    parser.add_argument('--visit', action='store_true', help='Visit Adaptation')
    parser.add_argument('--llm', action='store_true', help='LLM')
    parser.add_argument('--cuda', type=int, help='cuda device', default=0)
    parser.add_argument('--nlayer', type=int, help='numberoflayer', default=2)
    parser.add_argument('--gradadapt', action='store_true', help='Enable Grad Adapt')




    # Parse the arguments
    args = parser.parse_args()
    return args
