elifeLibrary {
    stage 'Checkout', {
        checkout scm
    }

    stage 'Build image', {
        sh 'docker build -t sciencebeam-gym .'
    }

    stage 'Run tests', {
        elifeLocalTests './project_tests.sh'
    }
}
