import sentiment

senti = sentiment.Sentiment()
senti.read_vocabulary()

def process_command(cmd, data):
    if cmd == "readReviews":
        if data == "file":
            senti.read_reviews_from_file()
        elif data == "cache":
            senti.read_reviews_from_cache()
    elif cmd == "trainBatch":
        iterations, iter_off = data.split(" ", 1)
        senti.train_batch(iterations=int(iterations), iteration_offset=int(iter_off))
    elif cmd == "trainSingle":
        type,sentence = data.split(" ", 1)
        if type == "pos":
            senti.train_single(sentence=sentence,expected_output=1)
        elif type == "neg":
            senti.train_single(sentence=sentence,expected_output=0)
        else:
            print("Unknown sentiment type")
    elif cmd == "loadModel":
        if data:
            senti.load_model(model_path=data)
        else:
            senti.load_model()
    elif cmd == "saveModel":
        senti.save_model(global_step=int(data))
    elif cmd == "testBatch":
        senti.test_batch()
    elif cmd == "testSingle":
        senti.test_single(data)
    else:
        print("Unknown command")
        print_commands()

def print_commands():
    print("-------------------------------------")
    print("Available commands")
    print("1. readReviews file")
    print("2. readReviews cache")
    print("3. trainBatch <iterations> <iterations-offset>")
    print("4. trainSingle pos <positive sentence>")
    print("5. trainSingle neg <negative sentence>")
    print("6. loadModel <optional-model-path>")
    print("7. saveModel <global-step-num>")
    print("8. testBatch")
    print("9. testSingle <Review text>")
    print("-------------------------------------")

print("Enter a command")
print_commands()
while True:
    usr_input = input().strip().split(' ', 1)
    if len(usr_input) == 2:
        cmd,data = usr_input
    elif len(usr_input) == 1:
        cmd,data = usr_input[0],None
    else:
        cmd,data = None,None
    process_command(cmd=cmd, data=data)
