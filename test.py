render_kwargs_train = {
        'network_query_fn':'network_query_fn',
        'perturb':'args.perturb',
        'N_importance':'args.N_importance',
        'network_fine':'model_fine',
        'N_samples':'args.N_samples',
        'network_fn':'model',
        'use_viewdirs':'args.use_viewdirs',
        'white_bkgd':'args.white_bkgd',
        'raw_noise_std':'args.raw_noise_std'
    }

render_kwargs_test = {k:render_kwargs_train[k] for k in render_kwargs_train}
print(f"==>> render_kwargs_test: {render_kwargs_test}")

for k in render_kwargs_train.keys():
    print(k)

    


