elifeLibrary {
    def commit
    stage 'Checkout', {
        checkout scm
        commit = elifeGitRevision()
    }

    stage 'Build image', {
        sh 'docker build -t elife/sciencebeam-gym .'
    }

    stage 'Run tests', {
        elifeLocalTests './project_tests.sh'
    }

    elifeMainlineOnly {
        stage 'Merge to master', {
            elifeGitMoveToBranch commit, 'master'
        }

        stage 'Downstream', {
            build job: '/dependencies/dependencies-sciencebeam-update-sciencebeam-gym', wait: false, parameters: [string(name: 'commit', value: commit)]
        }
    }
}
